import torch
from transformers import AutoTokenizer
import argparse
from dream.modeling_dream import DreamModel
import time
import re
import os
import warnings
import sys
import logging

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
CLEAR = '\033[2J\033[H' 


def clear_screen():
    """Clear the terminal screen"""
    print(CLEAR, end='')


def print_colored(text, color=RESET, bold=False):
    """Print colored text"""
    style = BOLD if bold else ''
    print(f"{style}{color}{text}{RESET}")


def main():
    """
    Demo script for Planned Diffusion generation with beautiful terminal output.
    
    Example
    -------
    python -m eval.demo
    python -m eval.demo --prompt "What is Aurora Borealis? Please be concise."
    python -m eval.demo --model_path dmisrael/ar-dream7b-sft-16ep
    """
    parser = argparse.ArgumentParser(description="Planned Diffusion Generation Demo")
    parser.add_argument("--model_path", type=str, default="dmisrael/planned-diffusion-dream7b-sft-16ep", 
                       help="Path to the model.")
    parser.add_argument("--prompt", type=str, default="What is Aurora Borealis? Please be concise.",
                       help="The prompt to generate text from.")
    parser.add_argument("--slow", action="store_true", default=False,
                       help="Whether to slow down the generation.")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Clear screen and show initial info
    clear_screen()
    print_colored("=" * 80, CYAN, bold=True)
    print_colored("PLANNED DIFFUSION GENERATION DEMO", CYAN, bold=True)
    print_colored("=" * 80, CYAN, bold=True)
    print()
    print_colored(f"Model: {args.model_path}", YELLOW)
    print_colored(f"Device: {device}", YELLOW)
    print()
    print_colored("-" * 80, CYAN)
    print()
    
    # Restore stderr for our prints
    sys.stderr.close()
    sys.stderr = original_stderr
    
    print_colored("Loading tokenizer...", MAGENTA)
    sys.stderr = open(os.devnull, 'w')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    sys.stderr.close()
    sys.stderr = original_stderr
    
    print_colored("Loading model...", MAGENTA)
    sys.stderr = open(os.devnull, 'w')
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=True,
    )
    model = model.to(device).eval()
    sys.stderr.close()
    sys.stderr = original_stderr
    
    # Build chat prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True
    ).to(device)
    
    # Decode the prompt for display
    prompt_text = tokenizer.decode(inputs[0], skip_special_tokens=False)
    # Track previous token ids to detect newly added/edited tokens per step
    prompt_len = inputs.shape[1]
    render_state = {"prev_ids": None, "ever_had_masks": False}
    
    def generation_tokens_hook_func(step, x, logits):
        """Hook function to display generation progress with colors"""
        
        # Clear screen and redraw everything
        clear_screen()
        print_colored("=" * 80, CYAN, bold=True)
        slow_str = "(SLOWED DOWN)" if args.slow else ""
        print_colored(f"PLANNED DIFFUSION GENERATION DEMO {slow_str}", CYAN, bold=True)
        print_colored("=" * 80, CYAN, bold=True)
        print()
        print(f"{YELLOW}{BOLD}GENERATION ({RED}Red{RESET}{YELLOW}{BOLD}=Prompt, {GREEN}Green{RESET}{YELLOW}{BOLD}=Autoregressive, {BLUE}Blue{RESET}{YELLOW}{BOLD}=Diffusion):{RESET}")
        print()
        
        # Decode current output
        current_output = tokenizer.decode(x[0].tolist(), skip_special_tokens=False)
        
        # Split into prompt and generation
        if tokenizer.eos_token in current_output:
            current_output = current_output.split(tokenizer.eos_token)[0]
        
        # Extract just the generation part (after the prompt)
        if len(current_output) > len(prompt_text):
            generated_part = current_output[len(prompt_text):]
        else:
            generated_part = ""
        
        # Display with proper colors: RED for prompt, GREEN for AR, BLUE for diffusion
        # Print prompt in red
        print(RED + prompt_text + RESET, end="", flush=True)
        
        # Token-wise rendering: highlight newly added/changed tokens in white for this step
        if generated_part:
            curr_ids = x[0].tolist()
            prev_ids = render_state["prev_ids"]

            changed_positions = set()
            if prev_ids is None:
                for idx in range(prompt_len, len(curr_ids)):
                    changed_positions.add(idx)
            else:
                max_len = max(len(prev_ids), len(curr_ids))
                for idx in range(prompt_len, max_len):
                    prev_tok = prev_ids[idx] if idx < len(prev_ids) else None
                    curr_tok = curr_ids[idx] if idx < len(curr_ids) else None
                    if prev_tok != curr_tok:
                        changed_positions.add(idx)

            # If we have EVER seen masks and now none remain, disable white highlighting
            mask_id = tokenizer.mask_token_id
            has_masks = (mask_id is not None) and (mask_id in curr_ids)
            if has_masks:
                render_state["ever_had_masks"] = True
            if (render_state["ever_had_masks"]) and (not has_masks):
                changed_positions.clear()

            in_async = False
            last_was_newline = False
            seen_any_async = False
            for ti in range(prompt_len, len(curr_ids)):
                piece = tokenizer.decode([curr_ids[ti]], skip_special_tokens=False)

                # Handle structural tags; async blocks open/close on newlines
                if piece == '<async>':
                    if not last_was_newline:
                        print()
                        last_was_newline = True
                    # Extra blank line before the first async block (after topics)
                    if not seen_any_async:
                        print()
                        last_was_newline = True
                    if ti in changed_positions:
                        print(WHITE + '<async>' + RESET, end="", flush=True)
                    else:
                        print(BLUE + '<async>' + BLUE, end="", flush=True)
                    last_was_newline = False
                    in_async = True
                    seen_any_async = True
                    continue
                if piece == '</async>':
                    if ti in changed_positions:
                        print(WHITE + '</async>' + RESET, end="", flush=True)
                    else:
                        print(BLUE + '</async>' + RESET, end="", flush=True)
                    print()
                    print()
                    last_was_newline = True
                    in_async = False
                    continue
                if piece == '<promise>-<topic>':
                    if ti in changed_positions:
                        print(WHITE + '<promise>-<topic>' + RESET, end="", flush=True)
                    else:
                        print(GREEN + '<promise>-<topic>' + RESET, end="", flush=True)
                    continue
                if piece == '</topic>':
                    if ti in changed_positions:
                        print(WHITE + '</topic>' + RESET, end="", flush=True)
                    else:
                        print(GREEN + '</topic>' + RESET, end="", flush=True)
                    continue
                if tokenizer.mask_token and piece == tokenizer.mask_token:
                    color = WHITE if ti in changed_positions else (BLUE if in_async else GREEN)
                    print(color + '[M]' + RESET, end="", flush=True)
                    last_was_newline = False
                    continue

                base_color = BLUE if in_async else GREEN
                color = WHITE if ti in changed_positions else base_color
                print(color + piece + RESET, end="", flush=True)
                last_was_newline = False

            render_state["prev_ids"] = curr_ids[:]
        
        # keep inline layout; no trailing blank line after each step
        
        if args.slow:
            time.sleep(0.3)
        return x
    
    # Start generation - the first hook call will display everything
    start_time = time.time()
    with torch.no_grad():
        output = model.planned_diffusion_generate(
            inputs,
            max_length=1024,
            steps_ratio=1.0,
            return_dict_in_generate=True,
            alg="pd_entropy",
            top_p=0.95,
            temperature=0.2,
            alg_temp=0.,
            generation_tokens_hook_func=generation_tokens_hook_func
        )
    end_time = time.time()
    
    print()
    time.sleep(1.5)
    
    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
    
    # Clean the text
    def clean_text(text):
        text = re.sub(r"(<async>)(\d+)", r"\1\n\2", text)
        regex = r'(\[PAD\]|<\/?async>|<sync\/>|<promise>.*?<\/topic>)'
        text = re.sub(regex, ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        text = text.replace("<|im_start|>assistant\n ", "<|im_start|>assistant\n")
        return text
    
    clean_generated_text = clean_text(generated_text)
    
    print_colored("CLEAN GENERATION:", YELLOW, bold=True)
    print(WHITE + clean_generated_text + RESET)
    print()
    print_colored("-" * 80, CYAN)
    print()
    time.sleep(3.0)
    
    tokens_generated = output.sequences.shape[1] - inputs.shape[1]
    generation_time = end_time - start_time
    tokens_per_second = tokens_generated / generation_time
    
    print_colored("STATISTICS:", YELLOW, bold=True)
    print_colored(f"  Time taken: {generation_time:.2f} seconds", CYAN)
    print_colored(f"  Tokens generated: {tokens_generated}", CYAN)
    print_colored(f"  Tokens per second: {tokens_per_second:.2f}", CYAN)
    print()
    print(f"Note: Demo is slower due to real-time color-coded printouts", end="")
    print()
    print()
    print_colored("=" * 80, CYAN, bold=True)


if __name__ == "__main__":
    main()
