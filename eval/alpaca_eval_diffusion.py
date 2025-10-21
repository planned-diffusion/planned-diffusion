import datasets
from transformers import AutoTokenizer, AutoModel
from dream.modeling_dream import DreamModel
from dream.control_tags import *
import torch
import torch.distributed as dist
import os
import argparse
import json
from tqdm import tqdm
import re   
import time
from datetime import timedelta

# Number of times to retry on Exception during generation
MAX_RETRY = 10

def clean_text(text):
    text = re.sub(r"(<async>)(\d+)", r"\1\n\2", text) # Add newlines between async and number
    regex = r'(\[PAD\]|<\/?async>|<sync\/>|<promise>.*?<\/topic>)'
    text = re.sub(regex, ' ', text)  # Replace all async, sync, and promise tags with spaces
    text = re.sub(r'[ \t]+', ' ', text) # Remove extra spaces
    text = text.strip()
    text = text.replace("<|im_start|>assistant\n ", "<|im_start|>assistant\n")
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text
    
def init_distributed():
    """Initialise torch.distributed if launched with torchrun."""
    if dist.is_available():
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        # If torchrun supplied multiple processes, WORLD_SIZE will be >1
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
            device_id = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(device_id)
            print(f"Initialized distributed process {dist.get_rank()} on device {device_id}")
            return dist.get_rank(), dist.get_world_size()
    return 0, 1


def evaluate(args):
    rank, world_size = init_distributed()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Determine the specific GPU for this process
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
    device_map = {"": device_id} if device_id is not None else None

    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
        use_cache=args.use_cache,
        disable_block_sparsity=args.disable_block_sparsity,
    )

    assert device_id is not None, "device_id is None"
    print(f"Moving model to device {device_id}")
    model.to(device_id)

    model.eval()

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    # Shard dataset across distributed processes for parallel evaluation
    if world_size > 1:
        eval_set = eval_set.shard(num_shards=world_size, index=rank)
    if args.num_samples is not None:
        eval_set = eval_set.select(range(args.num_samples))
    
    # Resume support: load existing shard and skip completed
    shard_path = f"{args.output_path}.part{rank}"
    if os.path.exists(shard_path):
        with open(shard_path, "r") as f:
            results = json.load(f)
    else:
        results = []
    completed_instructions = set(r["instruction"] for r in results)

    for example in tqdm(eval_set, disable=(rank != 0)):
        
        # Skip if already completed in previous run
        if example["instruction"] in completed_instructions:
            continue
        
        # set random seed
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["instruction"]}]
    
        # This matches the training format: apply_chat_template for user message + add_generation_prompt
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True
        ).to(model.device)
        prompt_len = inputs.shape[1]
        success = False
        last_exception = None
        max_length = args.max_length
        for _attempt in range(MAX_RETRY):
            try:
                start_time = time.time()
                # Generate the sample
                with torch.no_grad():
                    if args.dream_baseline:
                        output = model.diffusion_generate(
                            inputs,
                            max_length=max_length,
                            steps_ratio=args.steps_ratio,
                            return_dict_in_generate=True,
                            alg="entropy" if args.confidence_threshold is None else "confidence_threshold",
                            temperature=0.2, # Standard Dream sampling params
                            top_p=0.95,
                            alg_temp=0.,
                            threshold=args.confidence_threshold,
                        )
                    else:
                        output = model.planned_diffusion_generate(
                            inputs,
                            max_length=args.max_length,
                            steps_ratio=args.steps_ratio,
                            return_dict_in_generate=True,
                            alg="pd_entropy" if args.confidence_threshold is None else "pd_confidence_threshold",
                            temperature=0.2, # Standard Dream sampling params
                            top_p=0.95,
                            alg_temp=0.,
                            threshold=args.confidence_threshold,
                            length_scale=args.length_scale,
                        )
                end_time = time.time()
                
                latency = end_time - start_time
                
                
                output_ids = output.sequences[:, prompt_len:]
            
                raw_output = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=False,
                )

                output = clean_text(raw_output)
                result_item = {
                    "instruction": example["instruction"],
                    "output": output,
                    "raw_output": raw_output,
                    "latency": latency,
                    "generator": "planned_diffusion_dream"
                }
                results.append(result_item)
                # Save immediately after each successful generation
                with open(shard_path, "w") as f:
                    json.dump(results, f, indent=2)
                success = True
                break
            except Exception as e:
                last_exception = e

        if not success:
            output = f"[ERROR]: {last_exception}"
            
            results.append({
            "instruction": example["instruction"],
            "output": output,
            "raw_output": None,
            "latency": None,
            "generator": "planned_diffusion_dream"})
            # Save immediately after recording an error
            with open(shard_path, "w") as f:
                json.dump(results, f, indent=2)

        
    # --------------------------------------------------
    # Save results: each rank writes its shard, rank-0 merges
    # --------------------------------------------------
    shard_path = f"{args.output_path}.part{rank}"
    with open(shard_path, "w") as f:
        json.dump(results, f, indent=2)

    # Synchronise so that all shards are written
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged_results = []
        for r in range(world_size):
            part_file = f"{args.output_path}.part{r}"
            if not os.path.exists(part_file):
                raise FileNotFoundError(part_file)
            with open(part_file, "r") as pf:
                merged_results.extend(json.load(pf))

        with open(args.output_path, "w") as f:
            json.dump(merged_results, f, indent=2)

        # Optionally clean up shard files
        for r in range(world_size):
            try:
                os.remove(f"{args.output_path}.part{r}")
            except OSError:
                pass
    # --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="alpaca_eval_results.json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--steps_ratio", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="[distributed] local rank passed from torchrun")
    parser.add_argument("--dream_baseline", action="store_true", help="Use default inference mode.")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--use_cache", action="store_true", help="Use cache for generation.")
    parser.add_argument("--length_scale", type=float, default=None, help="Scale factor for async block lengths (None to use default 10)")
    parser.add_argument("--disable_block_sparsity", action="store_true", help="Disable block sparsity; use dense attention within generated blocks.")
    args = parser.parse_args()

    evaluate(args)

if __name__ == "__main__":
    main()
