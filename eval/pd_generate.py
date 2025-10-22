import torch
from transformers import AutoTokenizer
import argparse
from dream.modeling_dream import DreamModel
from dream.control_tags import *
import time
import re

def main():
    """
    Example
    -------
    python -m eval.pd_generate \\
        --model_path path/to/model \\
        --prompt "What is Aurora Borealis? Please be concise." \\
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="dmisrael/planned-diffusion-dream7b-sft-16ep", help="Path to the base model.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (ignored if do_sample=False)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--steps_ratio", type=float, default=1.0, help="Number of steps to generate.")
    parser.add_argument("--confidence_threshold", type=float, default=None, help="Confidence threshold for confidence threshold algorithm.")
    parser.add_argument("--use_cache", action="store_true", help="Use cache for generation.")
    parser.add_argument("--length_scale", type=float, default=None, help="Scale factor for async block lengths (None to use default 10).")
    parser.add_argument("--disable_block_sparsity", action="store_true", help="Disable block sparsity; use dense attention within generated blocks.")
    args = parser.parse_args()
    
    
    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16 if args.use_fp16 else torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=args.use_cache,
        disable_block_sparsity=args.disable_block_sparsity
    )

    model = model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True
    ).to(device)

    
    def generation_tokens_hook_func(step, x, logits):
        print(f"############ Step {step} ############")
        print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, "[M]"))
        return x
    
    
    start_time = time.time()
    with torch.no_grad():
        
        output = model.planned_diffusion_generate(
            inputs,
            max_length=args.max_length,
            steps_ratio=args.steps_ratio,
            return_dict_in_generate=True,
            alg="pd_confidence_threshold" if args.confidence_threshold is not None else "pd_entropy",
            threshold=args.confidence_threshold,
            top_p=args.top_p,
            temperature=args.temperature,
            alg_temp=0.,
            length_scale=args.length_scale,
            generation_tokens_hook_func=generation_tokens_hook_func
        )
        
    end_time = time.time()
    tokens_generated = output.sequences.shape[1] - inputs.shape[1]
    
    output_ids = output.sequences
    
    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=False,
    )
    
    def clean_text(text):
        text = re.sub(r"(<async>)(\d+)", r"\1\n\2", text) 
        regex = r'(\[PAD\]|<\/?async>|<sync\/>|<promise>.*?<\/topic>)'
        text = re.sub(regex, ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        text = text.replace("<|im_start|>assistant\n ", "<|im_start|>assistant\n")
        return text
    
    clean_generated_text = clean_text(generated_text)
    

    print("============ Clean Generation ============\n")
    print(clean_generated_text)
    
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Tokens per second: {tokens_generated / (end_time - start_time)}")
    


if __name__ == "__main__":
    main() 