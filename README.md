## Planned Diffusion

This repository contains the Planned Diffusion model code and evaluation scripts. Use this guide to set up a reproducible environment named `pd-env` and run the demo and AlpacaEval scripts.

### Environment setup (conda)

```bash
# Create and activate environment
conda create -y -n pd-env python=3.10
conda activate pd-env

# (Optional) Install CUDA toolkit if needed for your system
# conda install -y -c nvidia cuda-toolkit=12.6

# Install Python deps (pin to original pasta-diffusion if desired)
pip install -U pip
pip install -r requirements.txt
```

### Quickstart (demo)

The demo showcases Planned Diffusion with color-coded terminal output.

```bash
# From repo root
python -m eval.demo \
  --model_path dmisrael/planned-diffusion-dream7b-sft-16ep \
  --prompt "What is Aurora Borealis? Please be concise."
```

- Flags:
  - `--model_path`: Hugging Face repo or local checkpoint path (default: `dmisrael/planned-diffusion-dream7b-sft-16ep`).
  - `--prompt`: Input prompt string.

### AlpacaEval (Autoregressive baseline)

Multi-GPU generation using manual AR decoding with KV cache. Launch with `torchrun`.

```bash
# Example: 4 GPUs
torchrun --nproc_per_node=4 eval/alpaca_eval_ar.py \
  --model_path /path/to/model_or_hf_repo \
  --output_path outputs/alpaca_eval_ar.json
```

Outputs a JSON with generated responses and latency per sample.

### AlpacaEval (Planned Diffusion)

Multi-GPU evaluation using `model.planned_diffusion_generate`. Launch with `torchrun`.

```bash
# Example: 4 GPUs
torchrun --nproc_per_node=4 eval/alpaca_eval_diffusion.py \
  --model_path /path/to/model_or_hf_repo \
  --output_path outputs/alpaca_eval_pd.json \
  --max_length 1024 \
  --steps_ratio 1.0 \
  --use_cache
```

- Key flags:
  - `--model_path` (required): checkpoint or HF repo id
  - `--num_samples` (optional): evaluate a subset
  - `--output_path` (default: `alpaca_eval_results.json`)
  - `--max_length` (default: 1024)
  - `--steps_ratio` (default: 1.0)
  - `--random_seed` (default: 42)
  - `--dream_baseline`: switch to Dream diffusion baseline instead of Planned Diffusion
  - `--confidence_threshold` (optional): enable confidence-threshold variants
  - `--use_cache`: enable KV cache where applicable
  - `--length_scale` (optional): scale factor for async block lengths (e.g., 10)
  - `--disable_block_sparsity`: use dense attention within generated blocks
