import os
import re
import glob
from dataclasses import dataclass, field
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
)
from dream.modeling_dream import DreamModel
import json

from train.sft_trainer import preprocess_dataset, dLLMSFTDataset, dLLMDataCollator, dLLMTrainer

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Dream-org/Dream-v0-Instruct-7B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="predibase/wordle-sft",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    local_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to local JSONL dataset file. If provided, takes precedence over dataset_name."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    autoregressive_only: bool = field(
        default=False,
        metadata={"help": "Use standard autoregressive SFT training instead of hybrid PASTA diffusion training."}
    )
    diffusion_only: bool = field(
        default=False,
        metadata={"help": "Use diffusion-only PASTA training (disable AR objective/guidance)."}
    )
    ablate_topic: bool = field(
        default=False,
        metadata={"help": "Replace <topic>...</topic> with generic 'topic N' while preserving numeric id."}
    )
    ablate_sync: bool = field(
        default=False,
        metadata={"help": "Remove all occurrences of <sync> tokens (e.g., <sync/>) in training data."}
    )
    checkpoint_dir: str = field(
        default=None,
        metadata={"help": "Parent directory to search for checkpoint-x directories. Will automatically resume from the latest checkpoint found."}
    )
    ablate_block_sparsity: bool = field(
        default=False,
        metadata={"help": "If True, use ablation block-sparsity attention mask algorithm (v3); otherwise use standard PASTA (v2)."}
    )

def find_latest_checkpoint(checkpoint_dir):
    """
    Recursively search for checkpoint-x directories and return the one with the largest x.
    
    Args:
        checkpoint_dir (str): Parent directory to search for checkpoints
        
    Returns:
        str or None: Path to the latest checkpoint directory, or None if none found
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None
    
    # Pattern to match checkpoint-x directories where x is an integer
    checkpoint_pattern = re.compile(r'^checkpoint-(\d+)$')
    latest_checkpoint = None
    max_step = -1
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(checkpoint_dir):
        for dir_name in dirs:
            match = checkpoint_pattern.match(dir_name)
            if match:
                step = int(match.group(1))
                if step > max_step:
                    max_step = step
                    latest_checkpoint = os.path.join(root, dir_name)
    
    return latest_checkpoint

def load_local_jsonl(file_path):
    """Load data from a local JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Validate mutually exclusive training modes
    if data_args.autoregressive_only and data_args.diffusion_only:
        raise ValueError("Only one of --autoregressive_only or --diffusion_only can be set.")

    # Automatically set resume_from_checkpoint if checkpoint_dir is provided
    if data_args.checkpoint_dir and not training_args.resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(data_args.checkpoint_dir)
        if latest_checkpoint:
            training_args.resume_from_checkpoint = latest_checkpoint
            print(f"Auto-detected latest checkpoint: {latest_checkpoint}")
        else:
            print(f"No checkpoint-x directories found in {data_args.checkpoint_dir}")
    elif training_args.resume_from_checkpoint:
        print(f"Using explicitly provided checkpoint: {training_args.resume_from_checkpoint}")

    # Create a unique run name
    model_name = model_args.model_name_or_path.split("/")[-1]
    
    if data_args.local_dataset_path:
        dataset_name = os.path.basename(data_args.local_dataset_path).replace('.jsonl', '')
    else:
        dataset_name = data_args.dataset_name.split("/")[-1]

    # Include training mode in run name
    if data_args.autoregressive_only:
        mode_suffix = "ar"
    elif data_args.diffusion_only:
        mode_suffix = "diff"
    else:
        mode_suffix = "pasta"
    run_name = (
        f"{model_name}_{dataset_name}_{mode_suffix}_"
        f"lr{training_args.learning_rate}_"
        f"bs{training_args.per_device_train_batch_size}_"
        f"gas{training_args.gradient_accumulation_steps}"
    )
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)

    if data_args.autoregressive_only:
        print("Training mode: Autoregressive SFT")
    elif data_args.diffusion_only:
        print("Training mode: Diffusion-only PASTA")
    else:
        print("Training mode: Hybrid PASTA Diffusion")
    print(f"Saving checkpoints to {training_args.output_dir}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    # Add PASTA special tokens to prevent subword tokenization
    pasta_special_tokens = [
        "<async>",
        "</async>", 
        "<promise>-<topic>",
        "</topic>",
        "<sync/>",
        "[PAD]"
    ]
    
    # Add new tokens to tokenizer
    num_added_tokens = tokenizer.add_tokens(pasta_special_tokens)
    print(f"Added {num_added_tokens} PASTA special tokens to tokenizer")
    
    # Load the model using local DreamModel class
    model = DreamModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Resize token embeddings to accommodate new tokens
    if num_added_tokens > 0 and len(model.get_input_embeddings().weight) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model token embeddings to {len(tokenizer)} tokens")
    
    # Print the trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


    # Load and preprocess dataset
    if data_args.local_dataset_path:
        print(f"Loading local dataset from: {data_args.local_dataset_path}")
        dataset = {"train": load_local_jsonl(data_args.local_dataset_path)}
    else:
        print(f"Loading HuggingFace dataset: {data_args.dataset_name}")
        dataset = load_dataset(data_args.dataset_name)
    
    # Preprocess dataset with autoregressive_only flag
    train_data, eval_data = preprocess_dataset(
        dataset['train'], 
        tokenizer, 
        data_args.max_seq_length,
        autoregressive_only=data_args.autoregressive_only,
        diffusion_only=data_args.diffusion_only,
        ablate_topic=data_args.ablate_topic,
        ablate_sync=data_args.ablate_sync
    )

    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    print("Mask token ID: ", tokenizer.mask_token_id)

    # Create datasets (using dLLMSFTDataset for both modes)
    train_dataset = dLLMSFTDataset(train_data, tokenizer, data_args.max_seq_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, data_args.max_seq_length, eval=True)
    
    # Initialize data collator with autoregressive_only flag
    data_collator = dLLMDataCollator(
        tokenizer=tokenizer,
        mask_token_id=tokenizer.mask_token_id,
        max_length=data_args.max_seq_length,
        autoregressive_only=data_args.autoregressive_only,
        diffusion_only=data_args.diffusion_only,
        ablate_block_sparsity=data_args.ablate_block_sparsity
    )

    # Initialize trainer with explicit training mode flag
    trainer = dLLMTrainer(
        autoregressive_only=data_args.autoregressive_only,  # Explicit flag
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    if data_args.autoregressive_only:
        print("Using dLLMTrainer for autoregressive training (explicit AR mode)")
    elif data_args.diffusion_only:
        print("Using dLLMTrainer for diffusion-only training (explicit diffusion mode)")
    else:
        print("Using dLLMTrainer for hybrid PASTA diffusion training (explicit diffusion mode)")

    # Train the model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save the final model to output_dir/model_final
    final_model_dir = os.path.join(training_args.output_dir, "model_final")
    os.makedirs(final_model_dir, exist_ok=True)
    
    print(f"Saving final model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Model and tokenizer saved to: {final_model_dir}")

if __name__ == "__main__":
    main() 