# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Modified from Dream repos: https://github.com/HKUNLP/Dream

import time
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Callable, List

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from .pd_utils import create_pd_inputs, update_attention_mask, invert_and_expand_attention_mask, block_unmask, block_unmask_confidence_threshold
from .control_tags import *
import math

logger = logging.get_logger(__name__)



def round_to_nearest_divisor(x, y):
    divisors = set()
    for i in range(1, int(math.sqrt(x)) + 1):
        if x % i == 0:
            divisors.add(i)
            divisors.add(x // i)
    closest_divisor = min(divisors, key=lambda d: (abs(y - d), -d))
    return closest_divisor


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    
@dataclass
class PlannedDiffusionModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scaffold: Optional[torch.LongTensor] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", None)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        
        # Planned Diffusion specific params
        self.block_size: int = kwargs.pop("block_size", 25)  # Fixed size for each async block
        self.planning_end_id = kwargs.pop("planning_end_id", 151645)  # Token to mark end of autoregressive generation
        self.steps_ratio: float = kwargs.pop("steps_ratio", 1.0)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False, strict=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)
                    
        if generation_config.max_new_tokens is None and generation_config.max_length is not None:
            generation_config.max_new_tokens = generation_config.max_length - input_ids_length

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id
                if generation_config.planning_end_id is None:
                    generation_config.planning_end_id = self.generation_config.planning_end_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)
        planning_end_token_tensor = _tensor_or_none(generation_config.planning_end_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor
        generation_config._planning_end_token_tensor = planning_end_token_tensor

    @torch.no_grad()
    def _ar_sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        max_length: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        generation_tokens_hook_func=None,
        generation_logits_hook_func=None,
    ) -> torch.LongTensor:

        i = 0
        
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]))
            attention_mask = attention_mask.to(input_ids.device).to(torch.bool)
            
        curr_attention_mask = attention_mask

        # Generation loop
        while True:
                
            curr_input_ids = input_ids
            inverted_attention_mask = invert_and_expand_attention_mask(curr_attention_mask, self.dtype)
            

            # Forward pass with KV caching
            outputs = self(
                curr_input_ids,
                attention_mask=inverted_attention_mask,
            )
            
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            if top_p is not None and top_p < 1:
                next_token_logits = top_p_logits(next_token_logits, top_p)
            if top_k is not None:
                next_token_logits = top_k_logits(next_token_logits, top_k)
                
            next_token_logits = generation_logits_hook_func(i, curr_input_ids, next_token_logits)
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            
            # Check if we should stop
            if input_ids.shape[1] >= max_length:
                return input_ids, None
            
            # Append next token to input_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            curr_attention_mask = update_attention_mask(curr_attention_mask, torch.ones(1, 1, 1))
            input_ids = generation_tokens_hook_func(i, input_ids, None)
            i += 1
            
            if next_tokens == EOS_TOKEN_ID or next_tokens == SYNC_TOKEN_ID:
                break
        
        assert input_ids[0, -1] == SYNC_TOKEN_ID or input_ids[0, -1] == EOS_TOKEN_ID, "Sync/EOS token not found at the end of the sequence"
        return input_ids, curr_attention_mask

    @torch.no_grad()
    def planned_diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle generation config and kwargs
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)
        
        # 2. Prepare inputs and special tokens
        assert inputs is not None, "Input tensor must be provided"
        assert inputs.dim() == 2, f"Expected 2D input tensor [batch_size, seq_len], got shape {inputs.shape}"
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        assert attention_mask is None, "attention masking and batched generation not supported yet"
        self._prepare_special_tokens(generation_config, device=device)
        
        # 3. Prepare max length
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        end_of_generation = False
        
        curr_attention_mask = None
        curr_input_ids = input_ids
        while not end_of_generation:
        
            # 4. Autoregressive generation until im_end token
            curr_input_ids, curr_attention_mask = self._ar_sample(
                curr_input_ids,
                curr_attention_mask,
                generation_config.max_length,
                generation_config.temperature,
                generation_config.top_p,
                generation_config.top_k,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func
            )
            
            if curr_attention_mask is None:
                break
            
            final_planning_input_id = curr_input_ids[:, -1]
            curr_input_ids = curr_input_ids[:, :-1]
            curr_attention_mask = curr_attention_mask[:, :-1, :-1]
            
            # Create block attention mask
            length_scale = kwargs.get("length_scale", None)
            disable_block_sparsity = getattr(self.config, "disable_block_sparsity", False)
            curr_input_ids, curr_attention_mask, num_promises, block_info = create_pd_inputs(
                input_ids=curr_input_ids,
                prev_attention_mask=curr_attention_mask,
                device=device,
                length_scale=length_scale,
                disable_block_sparsity=disable_block_sparsity,
            )
            
            if num_promises == 0:
                break
        

            if "pd" not in generation_config.alg:
                max_block_size = (curr_input_ids == MASK_TOKEN_ID).sum().item()
            else:
                max_block_size = max(block_size for _, _, block_size in block_info) if block_info else 10
            
            # rounded_divisor = round_to_nearest_divisor(max_block_size, 1.0 / generation_config.steps_ratio)
            # rounded_steps = int(max_block_size / rounded_divisor)
            # generation_config.steps = rounded_steps
            
            generation_config.steps = int(max_block_size * generation_config.steps_ratio)

            ar_template = curr_input_ids.clone()
            
            # 6. Run diffusion over async blocks
            result = self._diff_sample(
                curr_input_ids,
                attention_mask=curr_attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
                format_inputs=False,
                threshold=kwargs.get("threshold", None),
                block_info=block_info
            )
            
            if final_planning_input_id.item() == SYNC_TOKEN_ID:
                end_of_generation = False
                curr_input_ids = result.sequences
                
                # add back the sync token to end of diffusion result
                curr_input_ids = torch.cat([curr_input_ids, final_planning_input_id.unsqueeze(0)], dim=-1)
                curr_attention_mask = update_attention_mask(curr_attention_mask, torch.ones(1, 1, 1))
                
            elif final_planning_input_id.item() == EOS_TOKEN_ID:
                end_of_generation = True
            else:
                raise ValueError(f"Unknown final input id: {final_planning_input_id.item()}")
        
        if curr_input_ids.shape[1] >= generation_config.max_length:
            curr_input_ids = curr_input_ids[:, :generation_config.max_length]
        
        if generation_config.return_dict_in_generate:
            return PlannedDiffusionModelOutput(
                sequences=curr_input_ids,
                scaffold=ar_template 
            )
        else:
            return curr_input_ids

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", 0.9)
        steps_ratio = kwargs.get("steps_ratio", 1.0)
        
        if generation_config.steps is None:
            generation_config.steps = int(generation_config.max_new_tokens * steps_ratio)

        result = self._diff_sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            threshold=threshold
        )
        return result

    def _diff_sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        threshold: Optional[float] = 0.9,
        format_inputs: bool = True,
        block_info: Optional[List[Tuple[int, int, int]]] = None
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
        
        if format_inputs:
            x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        else:
            x = input_ids

        if attention_mask is not None:
            
            if attention_mask.dim() == 2:
                # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # we do not mask the [MASK] tokens so value = 1.0
                attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[-1]), value=1.0)
                tok_idx = attention_mask.long().cumsum(-1) - 1
                tok_idx.masked_fill_(attention_mask == 0, 1)
                # attention_mask is of shape [B, 1, 1, N]
                # broadcast to [B, 1, N, N]
                attention_mask = torch.logical_and(
                    attention_mask.unsqueeze(-2),
                    attention_mask.unsqueeze(-1),
                )
            elif attention_mask.dim() == 3:
                # [batch_size, num_heads, seq_len, seq_len] - already in correct format
                # Assume attention_mask is correctly formatted if it is 4D
                tok_idx = None
                attention_mask = invert_and_expand_attention_mask(attention_mask, self.dtype)
            else:
                raise ValueError(f"Unexpected attention mask dimension: {attention_mask.dim()}")
            
        else:
            tok_idx = None
            attention_mask = "full"

        
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        i = 0
        if alg == 'confidence_threshold':
            mask_index = (x == mask_token_id)

            # assert mask_index.sum() % steps == 0, "mask_index.sum() must be divisible by steps"
            assert x.shape[0] == 1, "batch size must be 1"

            number_transfer_tokens = mask_index.sum().item() // steps
            left_tokens_last_step = 0

        if alg == 'pd_confidence_threshold':
            left_tokens_last_step_per_block = None
            # Calculate per-block transfer tokens once at the start
            mask_index = (x == mask_token_id)
            number_transfer_tokens_per_block = []
            for block_idx, (start_idx, end_idx, block_size) in enumerate(block_info):
                block_mask = mask_index[0, start_idx:end_idx]
                # number_transfer_tokens_per_block.append(max(1, int(block_mask.sum().item()) // steps))
                # number_transfer_tokens_per_block.append(max(1, math.ceil(block_mask.sum().item() / steps))) # Rounding up instead of down
                number_transfer_tokens_per_block.append(max(1, block_mask.sum().item() // steps))
                
        og_num_steps = steps
        
        while i < steps:
            mask_index = (x == mask_token_id)
            
            logits = self(input_ids=x, attention_mask=attention_mask, position_ids=tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1) # Shift logits
            
            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            if not alg == 'confidence_threshold' and not alg == 'pd_confidence_threshold':
                t = timesteps[i]
                s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            elif alg == 'confidence_threshold':
                number_transfer_tokens = mask_index.sum().item() // (og_num_steps - i) if i < og_num_steps else 0 # Add this for non-divisible steps
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                current_transfer_tokens = number_transfer_tokens + left_tokens_last_step
                
                current_transfer_tokens = max(1, current_transfer_tokens)
                current_transfer_tokens = min(current_transfer_tokens, mask_index.sum().item())
                
                left_tokens_last_step = 0
                selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                select_index = select_index.to(x.device)
                transfer_index[0, select_index[0]] = True
                selected = 0
                for k in range(1, current_transfer_tokens):
                    if selected_confidence[0, k] < threshold:
                        selected+=1
                        if i < steps - 1:
                            left_tokens_last_step += 1
                            transfer_index[0, select_index[0, k]] = False
                        else:
                            number_transfer_tokens = 0
                            steps += 1
                            left_tokens_last_step += 1
                            transfer_index[0, select_index[0, k]] = False
                
                x[transfer_index] = x_[transfer_index].clone()
                
            elif 'pd' in alg:
                assert block_info is not None, "block_info must be provided for Planned Diffusion algorithms"
                if alg == 'pd_maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    x = block_unmask(logits, mask_index, x, block_info, confidence, x0, t, s, steps, i, alg_temp)
                elif alg == 'pd_topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                    x = block_unmask(logits, mask_index, x, block_info, confidence, x0, t, s, steps, i, alg_temp)
                    
                elif alg == 'pd_entropy':
                     confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                     x = block_unmask(logits, mask_index, x, block_info, confidence, x0, t, s, steps, i, alg_temp)
                
                elif alg == 'pd_confidence_threshold':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    x, steps, left_tokens_last_step_per_block, number_transfer_tokens_per_block = block_unmask_confidence_threshold(
                        logits, mask_index, x, block_info, confidence, x0, steps, i, alg_temp, threshold,
                        left_tokens_last_step_per_block, number_transfer_tokens_per_block, og_num_steps)

                else:
                    raise RuntimeError(f"Unknown algorithm: {alg}")
                
                
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
            i += 1
            
            if (x == mask_token_id).sum() == 0:
                break
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x