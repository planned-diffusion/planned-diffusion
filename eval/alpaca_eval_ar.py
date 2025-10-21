#!/usr/bin/env python3
"""
Multi-GPU AlpacaEval Generation Script for Dream Model

This script generates responses to AlpacaEval prompts using multiple GPUs with torchrun.
It loads the AlpacaEval dataset from HuggingFace and generates responses using the Dream model.

Usage:
    torchrun --nproc_per_node=4 eval/alpaca_eval_generate.py \
        --model_path /path/to/dream/model \
        --output_path alpaca_eval_outputs.json \
        --max_new_tokens 512 \
        --temperature 0.7 \
        --top_p 0.9

Example:
    torchrun --nproc_per_node=4 eval/alpaca_eval_generate.py \
        --model_path /home/tianjin/orcd/pool/pasta-diffusion-data/output/ar-sft-ddp-v1.8/Dream-v0-Instruct-7B_pasta-training-100k-reordered_ar_lr5e-05_bs1_gas2/model_final \
        --output_path dream_alpaca_eval_outputs.json \
        --max_new_tokens 512
"""

import os
import sys

# Ensure the project root is on the Python path so that `import dream` works when
# this script is executed from the `scripts/` directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from dream.modeling_dream import DreamModel
from datasets import load_dataset
import argparse
import json
from typing import List, Dict, Any
import time
from tqdm import tqdm


def setup_distributed():
    """Initialize distributed training"""
    assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ, "RANK and WORLD_SIZE must be set"
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def manual_generate(model, tokenizer, input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, device="cuda"):
    """Manual autoregressive generation using model.forward() - optimized version"""
    import torch.nn.functional as F
    
    model.eval()
    generated_ids = input_ids.clone()
    attention_mask = torch.ones_like(generated_ids, dtype=torch.bool, device=device)

    # Throughput statistics
    gen_start_time = time.time()
    tokens_generated = 0

    # First forward pass with `use_cache=True` to prime the KV cache
    with torch.no_grad():
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            use_cache=True,
            is_causal=True,
        )

    if hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs

    # Validate that KV cache is present. If not, raise an explicit error to
    # avoid silently running full-sequence decoding every step.
    if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
        raise RuntimeError(
            "Model did not return `past_key_values` despite `use_cache=True`. "
            "Please ensure the checkpoint supports KV caching."
        )

    past_key_values = outputs.past_key_values

    # Prepare special stop tokens
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Continue autoregressive generation token-by-token, leveraging the KV cache
    for step in range(max_new_tokens):
        next_token_logits = logits[0, -1, :]

        # Apply temperature scaling
        if do_sample and temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-p nucleus sampling
        if do_sample and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("Inf")

        # Sample / greedy select next token
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Stop if EOS or <|im_end|>
        if next_token.item() == tokenizer.eos_token_id or next_token.item() == im_end_token_id:
            break

        # Append token to sequence / attention mask
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=torch.bool, device=device)
        ], dim=-1)

        tokens_generated += 1
        if tokens_generated % 20 == 0:
            elapsed = time.time() - gen_start_time
            tps = tokens_generated / elapsed if elapsed > 0 else 0
            print(f"    ↳ manual_generate: {tokens_generated} tokens in {elapsed:.2f}s ({tps:.2f} tok/s)", flush=True)

        # Forward pass **only** the newly generated token with cached KV
        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                is_causal=True,
            )

        # Sanity check: ensure cache grows as expected
        if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
            raise RuntimeError("KV cache unexpectedly missing on incremental step.")

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else past_key_values

    # Final throughput summary
    total_elapsed = time.time() - gen_start_time
    if tokens_generated > 0:
        print(f"    ↳ manual_generate DONE: {tokens_generated} tokens, {total_elapsed:.2f}s, {(tokens_generated/total_elapsed):.2f} tok/s", flush=True)

    return generated_ids, total_elapsed


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int, 
                     temperature: float, top_p: float, device: str, max_tokens: int = None) -> str:
    """Generate a single response using the Dream model.

    If ``max_tokens`` is provided, the effective ``max_new_tokens`` will be
    computed dynamically as ``max_tokens - prompt_tokens`` for each prompt.
    """
    
    # Format instruction using ChatML format (same as training)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True
    ).to(device)
    
    # Determine effective max_new_tokens based on total token budget
    effective_max_new_tokens = max_new_tokens
    if max_tokens is not None:
        prompt_tokens = inputs.shape[1]
        effective_max_new_tokens = max(0, max_tokens - prompt_tokens)

    # Generate response using manual generation
    with torch.no_grad():
        outputs, total_elapsed = manual_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=inputs,
            max_new_tokens=effective_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            device=device
        )
    
    # Decode only the new tokens (response part)
    response_tokens = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    return response, total_elapsed


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU AlpacaEval generation with Dream model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the Dream model checkpoint")
    parser.add_argument("--output_path", type=str, default="alpaca_eval_outputs.json",
                       help="Output file path for generated responses")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Total token budget (prompt + generated). Overrides max_new_tokens per sample.")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    # Batch size is no longer needed because we iterate one sample at a time
    # over the dataset indices assigned to the current rank.
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use float16 precision")
    
    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    
    if rank == 0:
        print(f"Starting AlpacaEval generation with {world_size} GPUs")
        print(f"Model: {args.model_path}")
        print(f"Output: {args.output_path}")
        if args.max_tokens is not None:
            print(f"Max tokens (prompt+gen): {args.max_tokens}")
        else:
            print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
    
    # Load AlpacaEval dataset
    if rank == 0:
        print("Loading AlpacaEval dataset...")
    
    dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")
    instructions = dataset["instruction"]
    
    if rank == 0:
        print(f"Loaded {len(instructions)} instructions")
    
    # Determine which instruction indices this GPU (rank) should process. We
    # simply shard by taking every `world_size`-th element, starting from the
    # current rank. This removes the need for `DistributedSampler` and a
    # `DataLoader`.
    assigned_indices = list(range(rank, len(instructions), world_size))
    if rank == 0:
        print(f"GPU {rank} will generate {len(assigned_indices)} / {len(instructions)} instructions")
    
    # Load model and tokenizer
    if rank == 0:
        print("Loading model and tokenizer...")
    
    # Dtype logic
    dtype = torch.float16 if args.use_fp16 else torch.bfloat16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load model
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    
    # No DistributedDataParallel; inference is done independently on each GPU
    model.eval()
    
    if rank == 0:
        print("Model loaded successfully")
        print("Starting generation...")
    
    # Generate responses
    all_outputs = []
    start_time = time.time()
    
    for local_idx, instruction_id in enumerate(tqdm(assigned_indices, desc=f"GPU {rank} generating", disable=rank!=0)):
        instruction = instructions[instruction_id]
        
        try:
            # Generate response
            response, total_elapsed = generate_response(
                model,
                tokenizer,
                instruction,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                device,
                args.max_tokens
            )
            
            # Create output entry
            output_entry = {
                "instruction": instruction,
                "output": response,
                "generator": f"dream_model_rank_{rank}",
                "instruction_id": instruction_id,
                "latency": total_elapsed
            }
            
            all_outputs.append(output_entry)

            # Stream the response to stdout for real-time monitoring
            print(f"[GPU {rank}] Instruction {instruction_id}: {response}", flush=True)
            
            if rank == 0 and (local_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                remaining_items = len(assigned_indices) - (local_idx + 1)
                eta = elapsed / (local_idx + 1) * remaining_items
                avg_time_per_item = elapsed / (local_idx + 1)
                print(f"Processed {local_idx + 1}/{len(assigned_indices)} items. Avg: {avg_time_per_item:.2f}s/item, ETA: {eta/60:.1f}min")
                
        except Exception as e:
            if rank == 0:
                print(f"Error processing instruction {instruction_id}: {e}")
            # Add empty response for failed generations
            output_entry = {
                "instruction": instruction,
                "output": f"ERROR: {str(e)}",
                "generator": f"dream_model_rank_{rank}",
                "instruction_id": instruction_id,
                "latency": total_elapsed
            }
            all_outputs.append(output_entry)
    
    # Gather all outputs from all GPUs
    if rank == 0:
        print("Gathering outputs from all GPUs...")
    
    # Convert to tensors for gathering (we'll gather the indices and reconstruct)
    local_output_data = []
    for output in all_outputs:
        local_output_data.append({
            'instruction_id': output['instruction_id'],
            'instruction': output['instruction'],
            'output': output['output'],
            'generator': 'dream_model',  # Unified generator name
            'latency': output['latency']
        })
    
    # Gather all outputs on rank 0
    if world_size > 1:
        gathered_outputs = [None] * world_size
        dist.gather_object(local_output_data, gathered_outputs if rank == 0 else None, dst=0)
        
        if rank == 0:
            # Flatten and sort by instruction_id
            all_gathered_outputs = []
            for rank_outputs in gathered_outputs:
                all_gathered_outputs.extend(rank_outputs)
            
            # Sort by instruction_id to maintain order
            all_gathered_outputs.sort(key=lambda x: x['instruction_id'])
            final_outputs = all_gathered_outputs
        else:
            final_outputs = []
    else:
        final_outputs = local_output_data
    
    # Save outputs on rank 0
    if rank == 0:
        print(f"Saving {len(final_outputs)} outputs to {args.output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_outputs, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        print(f"Generation completed in {total_time:.1f} seconds")
        print(f"Average time per sample: {total_time/len(final_outputs):.2f} seconds")
        print(f"Outputs saved to: {args.output_path}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
