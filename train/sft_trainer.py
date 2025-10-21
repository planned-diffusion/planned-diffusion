# SFT code from https://github.com/dllm-reasoning/d1/blob/main/SFT/sft_trainer.py 

import torch
import re
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
from .attention_mask_v2 import compute_pasta_diffusion_metadata_v2, special_tokens_from_tokenizer
from .attention_mask_v3 import compute_pasta_diffusion_metadata_ablate_block_sparsity, special_tokens_from_tokenizer
from .pad import pad


class dLLMTrainer(Trainer):
    def __init__(self, autoregressive_only=False, *args, **kwargs):
        """
        Initialize dLLMTrainer with explicit training mode flag
        
        Args:
            autoregressive_only (bool): If True, use autoregressive training objective.
                                      If False, use diffusion training objective.
        """
        super().__init__(*args, **kwargs)
        self.autoregressive_only = autoregressive_only
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute loss based on the explicit training mode
        """
        if self.autoregressive_only:
            return self._compute_autoregressive_loss(model, inputs, num_items_in_batch, return_outputs)
        else:
            return self._compute_diffusion_loss(model, inputs, num_items_in_batch, return_outputs)
    
    def _compute_autoregressive_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute loss for standard autoregressive training
        """
        # print("Computing autoregressive loss")
        labels = inputs.pop("labels")
        num_prompt_tokens = inputs.pop("num_prompt_tokens")
        # Remove 't' if present (not needed for AR mode)
        inputs.pop("t", None)
        
        outputs = model(**inputs)
        logits = outputs.logits

        # shift labels/logits by 1
        labels = labels[:, 1:]
        logits = logits[:, :-1, :]
        
        input_ids = inputs["input_ids"]
        # print("Last 5 input_ids:", input_ids[0][num_prompt_tokens:num_prompt_tokens+10])
        # print("Last 5 labels:", labels[0][num_prompt_tokens:num_prompt_tokens+10])
        # print("All labels:", labels[0])
        # print("Last 5 logits:", torch.argmax(logits, dim=-1)[0][:10])
        
        # Compute standard cross-entropy loss
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        
        # Log unscaled loss for monitoring
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({
                "unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item(),
                "training_mode": "autoregressive"
            })
        
        # Autoregressive mode: use standard loss normalization
        # Only compute loss on non-ignored tokens (labels != -100)
        valid_tokens = (labels != -100)
        loss = (unscaled_loss * valid_tokens).sum() / valid_tokens.sum()
        
        return loss if not return_outputs else (loss, outputs)
    
    def _compute_diffusion_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute loss for PASTA diffusion training
        """
        labels = inputs.pop("labels")
        t = inputs.pop("t")  # Required for diffusion mode
        num_prompt_tokens = inputs.pop("num_prompt_tokens")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute standard cross-entropy loss
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        
        # Log unscaled loss for monitoring
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({
                "unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item(),
                "training_mode": "diffusion"
            })
        
        # Diffusion mode: scale loss by timestep
        loss = unscaled_loss / t
        # Normalize by total tokens minus prompt tokens
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Unified data collator supporting both:
    1. Standard autoregressive SFT training (autoregressive_only=True)
    2. Hybrid autoregressive/diffusion training with PASTA (autoregressive_only=False)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        self.autoregressive_only = kwargs.get("autoregressive_only", False)
        self.diffusion_only = kwargs.get("diffusion_only", False)
        self.ablate_block_sparsity = kwargs.get("ablate_block_sparsity", False)
        if self.ablate_block_sparsity:
            print("================================================")
            print("Ablating block sparsity")
            print("================================================")

        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
        
        # Initialize PASTA special tokens
        self.special_tokens = special_tokens_from_tokenizer(self.tokenizer)

    def forward_process(self, batch, ar_mask, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        # Only apply noise to diffusion tokens (ar_mask == 0)
        mask_indices = (torch.rand((B, N), device=input_ids.device) < t) & (ar_mask == 0)
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def compute_pasta_attention_mask(self, input_ids):
        """
        Compute PASTA attention mask and labels for a single sequence (batch_size=1, no padding)
        """

        assert input_ids.shape[0] == 1, f"Expected batch_size=1, got {input_ids.shape[0]}"
        token_ids = input_ids[0].tolist()
        
        # Compute PASTA metadata (gate by ablation flag)
        if self.ablate_block_sparsity:
            mask, pasta_labels, ar_mask = compute_pasta_diffusion_metadata_ablate_block_sparsity(
                token_ids, 
                self.special_tokens
            )
        else:
            mask, pasta_labels, ar_mask = compute_pasta_diffusion_metadata_v2(
                token_ids, 
                self.special_tokens
            )
        
        # Convert to tensor to allow for boolean masking
        pasta_labels = torch.tensor(pasta_labels)

        # Change <async>, </async> labels to -100
        pasta_labels[pasta_labels == self.special_tokens.async_start] = -100
        pasta_labels[pasta_labels == self.special_tokens.async_end] = -100
        
        # Ensure labels length matches input length by right padding/truncating
        input_len = input_ids.shape[1]
        if pasta_labels.shape[0] < input_len:
            pad_len = input_len - pasta_labels.shape[0]
            pasta_labels = F.pad(pasta_labels, (0, pad_len), value=-100)
        elif pasta_labels.shape[0] > input_len:
            pasta_labels = pasta_labels[:input_len]
        
        return mask.unsqueeze(0), pasta_labels.unsqueeze(0), ar_mask.unsqueeze(0)

    def __call__(self, batch):
        if self.autoregressive_only:
            # ========== AUTOREGRESSIVE SFT MODE ==========
            # Assert batch size = 1 for simplicity
            assert len(batch) == 1, f"Autoregressive mode requires batch_size=1, got {len(batch)}"            
            batch = super().__call__(batch)
            
            # For autoregressive training: just copy input_ids to labels
            # The model will handle causal shifting automatically
            batch["labels"] = batch["input_ids"].clone()
            
            # Handle prompt masking if prompt_lengths is provided
            if "prompt_lengths" in batch:
                prompt_lengths = batch.pop("prompt_lengths")
                prompt_length_indices = torch.arange(batch["input_ids"].shape[1]).unsqueeze(0)
                prompt_mask = prompt_length_indices < prompt_lengths
                batch["labels"][prompt_mask] = -100  # Don't compute loss on prompt tokens
                batch["num_prompt_tokens"] = prompt_mask.sum()
            else:
                batch["num_prompt_tokens"] = torch.tensor(0, device=batch["input_ids"].device)
            
            # Don't set attention_mask - model uses default causal mask
            # Set t=None for autoregressive mode (will be handled by trainer)
            batch["t"] = None
            attention_mask = torch.tril(torch.ones(batch["input_ids"].shape[1], batch["input_ids"].shape[1])).unsqueeze(0).unsqueeze(0)
            attention_bias = torch.where(
                attention_mask == 0,
                torch.tensor(float('-inf'), dtype=torch.bfloat16, device=attention_mask.device),
                torch.tensor(0.0, dtype=torch.bfloat16, device=attention_mask.device)
            )
            batch["attention_mask"] = attention_bias
            
        else:
            # ========== HYBRID PASTA DIFFUSION MODE ==========
            batch = super().__call__(batch)
            
            assert batch["input_ids"].shape[0] == 1, f"Expected batch_size=1, got {batch['input_ids'].shape[0]}"
            batch["attention_mask"], batch["labels"], ar_mask = self.compute_pasta_attention_mask(batch["input_ids"])
            if self.diffusion_only:
                batch["attention_mask"] = torch.ones_like(batch["attention_mask"])
                batch["labels"] = batch["input_ids"].clone()[:, 1:]
                batch["labels"] = F.pad(batch["labels"], (0, batch["input_ids"].shape[1] - batch["labels"].shape[1]), value=-100)
                ar_mask = torch.zeros_like(ar_mask)
            
            # assert labels have the same length as input_ids
            if batch["labels"].shape[1] != batch["input_ids"].shape[1]:
                print("\n=== Length Mismatch Debug Info ===")
                print(f"Labels shape: {batch['labels'].shape}")
                print(f"Input IDs shape: {batch['input_ids'].shape}")
                print("\nLabels content:")
                print(batch["labels"][0].tolist())
                print("\nInput IDs content:")
                print(batch["input_ids"][0].tolist())
                print("\nDetokenized labels:")
                safe_label_ids = [tid for tid in batch["labels"][0].tolist() if isinstance(tid, int) and tid >= 0]
                print(self.tokenizer.decode(safe_label_ids))
                print("\nDetokenized input IDs:")
                print(self.tokenizer.decode(batch["input_ids"][0].tolist()))
                print("===============================\n")
                raise AssertionError(f"Labels and input_ids have different lengths: {batch['labels'].shape[1]} != {batch['input_ids'].shape[1]}")

            # Convert binary attention mask to attention bias
            # 0 in mask means -inf bias (no attention), 1 means 0 bias (full attention)
            attention_bias = torch.where(
                batch["attention_mask"] == 0,
                torch.tensor(float('-inf'), dtype=torch.bfloat16, device=batch["attention_mask"].device),
                torch.tensor(0.0, dtype=torch.bfloat16, device=batch["attention_mask"].device)
            )
            batch["attention_mask"] = attention_bias

            noisy_batch, batch["t"], mask_indices = self.forward_process(batch, ar_mask)
            batch["labels"][~(mask_indices | (ar_mask == 1))] = -100

            # Assert that mask_indices and ar_mask are disjoint (no overlap)
            assert not torch.any(mask_indices & (ar_mask == 1)), "mask_indices and ar_mask should be disjoint"
            
            assert "prompt_lengths" in batch, "prompt_lengths must be in batch"
            if "prompt_lengths" in batch:
                # Minus one because next token pred label is shifted by one
                prompt_lengths = batch.pop("prompt_lengths") - 1
                prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
                prompt_mask = prompt_length_indices < prompt_lengths
                noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
                batch["labels"][prompt_mask] = -100
                batch["num_prompt_tokens"] = prompt_mask.sum()
            batch["input_ids"] = noisy_batch.long()
                
        return batch


def is_valid_sample(sample, completion_tokens):
    """Sophisticated validation using tokenized completion and stack-based async tracking"""
    if not sample.get("prompt") or not sample.get("completion"):
        return False
    
    # Track async block state - but reject nested blocks
    in_async_block = False
    
    for token in completion_tokens:
        if token == "<async>":
            # Reject nested async blocks
            if in_async_block:
                return False
            in_async_block = True
        elif token == "</async>":
            # Found closing tag without opening tag
            if not in_async_block:
                return False
            in_async_block = False
    
    # Valid if we're not currently inside an async block
    return not in_async_block


def preprocess_dataset(data, tokenizer, max_length, test_split=0.1, autoregressive_only=False, diffusion_only=False, ablate_topic=False, ablate_sync=False):
    preprocessed_data = []
    total_samples = len(data)
    valid_samples = 0
    
    # Initialize special tokens for padding
    special_tokens = special_tokens_from_tokenizer(tokenizer)
    
    for i in tqdm(range(total_samples), desc="Preprocessing dataset"):
        completion_tokens = tokenizer.tokenize(data[i]["completion"])
        if not is_valid_sample(data[i], completion_tokens):
            continue
            
        prompt = [{"role": "user", "content": data[i]["prompt"]}]
        response = [{"role": "assistant", "content": data[i]["completion"]}]

        # Optionally ablate topic content to a generic placeholder while preserving numeric suffix
        if ablate_topic:
            response[0]["content"] = re.sub(
                r"<topic>.*?(\d+)</topic>",
                r"<topic>topic \1</topic>",
                response[0]["content"],
                flags=re.DOTALL,
            )

        # Optionally ablate sync tokens by removing them entirely
        if ablate_sync:
            # Remove both self-closing and non-self-closing variants just in case
            response[0]["content"] = response[0]["content"].replace("<sync/>", "")

        should_clean = autoregressive_only or diffusion_only
        if should_clean:
            # Clean special tokens.
            response[0]["content"] = re.sub(r"<promise>-<topic>.*?</topic>", "", response[0]["content"])
            response[0]["content"] = response[0]["content"].replace("<async>", "")
            response[0]["content"] = response[0]["content"].replace("</async>", "")
            response[0]["content"] = response[0]["content"].replace("<sync/>", "")
        
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        padding = False
        if diffusion_only:
            padding = "max_length"
        if should_clean:
            # Simple tokenization for autoregressive mode
            tokenized_input = tokenizer(
                inputs, return_tensors="pt", truncation=True, max_length=max_length, padding=padding
            ).input_ids.squeeze(0)
            
            if len(tokenized_input) > max_length:
                continue
                
            padded_tokens = tokenized_input
            
        else:
            # Original PASTA preprocessing with async padding for hybrid mode
            # First tokenize without padding to get token IDs
            tokenized_input = tokenizer(
                inputs, return_tensors="pt", truncation=False, padding=False
            ).input_ids.squeeze(0).tolist()
            
            # Trim tokens after the last <|im_end|>
            im_end_id = special_tokens.eos
            
            if im_end_id in tokenized_input:
                last_im_end_index = len(tokenized_input) - 1 - tokenized_input[::-1].index(im_end_id)
                tokenized_input = tokenized_input[:last_im_end_index + 1]
            
            # Apply stochastic padding to async blocks
            padded_tokens = pad(
                tokenized_input, 
                mode="stochastic",
                special_tokens=special_tokens,
                padding_low=0,
                padding_high=11  # 11 because range is exclusive
            )
            
            # Now check if padded sequence fits within max_length
            if len(padded_tokens) >= max_length:
                # Drop this example to avoid potential PASTA label mismatches
                continue
                
            # Convert back to tensor
            padded_tokens = torch.tensor(padded_tokens, dtype=torch.long)
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": padded_tokens,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )
        valid_samples += 1

    # Print simple summary
    valid_fraction = valid_samples / total_samples if total_samples > 0 else 0
    mode_str = "autoregressive" if autoregressive_only else ("diffusion_only" if diffusion_only else "hybrid")
    print(f"\nDataset preprocessing complete ({mode_str} mode):")
    print(f"Total samples: {total_samples:,}")
    print(f"Valid samples: {valid_samples:,}")
    print(f"Valid fraction: {valid_fraction:.3f}")

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data