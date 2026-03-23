# AeroRL

Production-oriented library for VLM-RL training runtime orchestration and performance benchmarking.

## What AeroRL delivers

- Unified runtime wiring for train/reference model roles
- Vision-token-aware loss path for multimodal RL batches
- Backend routing for `trl` / `verl` environments
- Quantized reference runtime selection
- Reproducible throughput and VRAM benchmark reports

## Why choose this

- Faster path from idea to measurable VLM-RL experiments
- Consistent API across experimentation and evaluation workflows
- Versioned benchmark artifacts for transparent performance tracking
- Test-covered core runtime behaviors

## Core API

- `wrap_vlm_for_rl(...)` — runtime assembly for train/reference roles
- `AeroRLTrainer` — training lifecycle hooks and masked train step
- `masked_cross_entropy_loss(...)` — vision-token-aware loss computation
- `create_quantized_reference_runtime(...)` — quantized reference runtime metadata

## Install

```bash
git clone https://github.com/sabdulmajid/aeroRL.git
cd aeroRL
python -m pip install -e .
```

## Quick Start

```bash
python -m pytest -q
python benchmarks/vlm_grpo_benchmark.py --mode real --model Qwen/Qwen2.5-VL-7B-Instruct --steps 20 --matrix-size 512
```

## Performance Snapshot

Latest benchmark artifacts:
- `reports/benchmark-real-2026-03-23.json`
- `reports/benchmark-matrix-real-2026-03-23.json`

| Benchmark | Mode | Steps | Key Result |
|---|---|---:|---|
| Single model (`Qwen/Qwen2.5-VL-7B-Instruct`) | Real | 20 | 146.5458 it/s, 0.0094 GB peak VRAM |
| Multi-model matrix (3 models) | Real | 40 | 0.0314 GB max peak VRAM |

Validation:
- `7 passed` tests

## Easy examples

### 1) Runtime wiring

```python
from aerorl import AeroRLConfig, wrap_vlm_for_rl

cfg = AeroRLConfig(quant_ref_bits=8, trainer_backend="auto")
train_rt, ref_rt = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", cfg)
print(train_rt["trainer"])
print(ref_rt["quantization_mode"])
```

### 2) One training step with vision masking

```python
import torch
from aerorl import AeroRLConfig, AeroRLTrainer

trainer = AeroRLTrainer(AeroRLConfig(mask_vision_tokens=True))
trainer.on_train_start()

logits = torch.randn(2, 4, 8, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
vision_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])

print(trainer.train_step(logits, labels, vision_mask))
print(trainer.on_train_end())
```

### 3) Matrix benchmark

```bash
python benchmarks/vlm_grpo_benchmark.py \
  --mode real --steps 40 --matrix-size 2048 \
  --models Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-1.5-7b-hf,microsoft/Phi-3-vision-128k-instruct
```

## Deployment note

Use `trainer_backend="trl"` or `trainer_backend="verl"` in `AeroRLConfig` when those packages are present in your environment.
