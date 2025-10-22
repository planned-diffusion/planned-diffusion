## Planned Diffusion

This is the official repository for Planned Diffusion.
> 
> arXiv: [Planned Diffusion](https://arxiv.org/abs/2510.18087)
> 
> Authors: Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, Michael Carbin
> 
> Abstract:
> A central challenge in large language model inference is the trade-off between generation speed and output quality. Autoregressive models produce high-quality text but generate tokens sequentially. Diffusion models can generate tokens in parallel but often need many iterations to match the same quality. We propose planned diffusion, a hybrid method that combines the strengths of both paradigms. Planned diffusion works in two stages: first, the model creates a short autoregressive plan that breaks the output into smaller, independent spans. Second, the model generates these spans simultaneously using diffusion. This approach expands the speed-quality Pareto frontier and provides a practical path to faster, high-quality text generation. On AlpacaEval, a suite of 805 instruction-following prompts, planned diffusion achieves Pareto-optimal trade-off between quality and latency, achieving 1.27x to 1.81x speedup over autoregressive generation with only 0.87\% to 5.4\% drop in win rate, respectively. Our sensitivity analysis shows that the planning mechanism of planned diffusion is minimal and reliable, and simple runtime knobs exist to provide flexible control of the quality-latency trade-off.

### Environment Setup

```bash
# Create and activate environment
conda create -y -n pd-env python=3.10
conda activate pd-env
pip install -U pip
pip install -r requirements.txt
```

### Quickstart

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

### Training

The training entrypoint is `train/train.py` (Hugging Face Trainer). It supports three modes via flags:
- Hybrid Planned Diffusion (default)
- Autoregressive-only: add `--autoregressive_only`
- Diffusion-only: add `--diffusion_only`

Single GPU example (Planned Diffusion):
```bash
python -m train.train \
  --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
  --dataset_name tatsu-lab/alpaca \
  --output_dir outputs \
```

Autoregressive-only SFT:
```bash
python -m train.train \
  --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
  --dataset_name tatsu-lab/alpaca \
  --output_dir outputs \
  --autoregressive_only 
```

Diffusion-only:
```bash
python -m train.train \
  --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
  --dataset_name tatsu-lab/alpaca \
  --output_dir outputs \
  --diffusion_only 
```

Training artifacts:

- Planned Diffusion Dream7B SFT (16 epochs): [dmisrael/planned-diffusion-dream7b-sft-16ep](https://huggingface.co/dmisrael/planned-diffusion-dream7b-sft-16ep)
- Planned Diffusion Dream7B SFT Full Attention (16 epochs): [dmisrael/planned-diffusion-dream7b-sft-full-attention-16ep](https://huggingface.co/dmisrael/planned-diffusion-dream7b-sft-full-attention-16ep)
- AR Dream7B SFT (16 epochs): [dmisrael/ar-dream7b-sft-16ep](https://huggingface.co/dmisrael/ar-dream7b-sft-16ep)

### Inference

`eval/pd_generate.py` provides a CLI for Planned Diffusion text generation.

```bash
# Basic usage
python -m eval.pd_generate \
  --model_path dmisrael/planned-diffusion-dream7b-sft-16ep \
  --prompt "What is Aurora Borealis? Please be concise."

# With options
python -m eval.pd_generate \
  --model_path dmisrael/planned-diffusion-dream7b-sft-16ep \
  --prompt "Explain diffusion vs autoregressive generation." \
  --max_length 2048 \
  --steps_ratio 1.0 \
  --temperature 0.2 \
  --top_p 0.95 \
  --use_cache \
  --disable_block_sparsity
```

### Evaluation

AlpacaEval results for baseline AR decoding with KV cache. Launch with `torchrun`.

```bash
# Example: 4 GPUs
torchrun --nproc_per_node=4 eval/alpaca_eval_ar.py \
  --model_path /path/to/model_or_hf_repo \
  --output_path outputs/alpaca_eval_ar.json
```

Outputs a JSON with generated responses and latency per sample.


AlpacaEval results for Planned Diffusion. Launch with `torchrun`.

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

### Citation

ArXiv: Planned Diffusion: https://arxiv.org/abs/2510.18087

```bibtex
@misc{israel2025planneddiffusion,
      title={Planned Diffusion}, 
      author={Daniel Israel and Tian Jin and Ellie Cheng and Guy Van den Broeck and Aditya Grover and Suvinay Subramanian and Michael Carbin},
      year={2025},
      eprint={2510.18087},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.18087}, 
}
```
