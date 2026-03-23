# AeroRL

AeroRL is a small library that helps you run VLM-RL experiments with less boilerplate and with measurable performance output.

## In one sentence

Give AeroRL your model setup and training tensors, and it gives you:
- runtime wiring (train + reference)
- masked loss for vision-language tokens
- training step lifecycle hooks
- benchmark reports (throughput + VRAM)
- composable reward stack + offline replay reward evaluation

## What problem it solves

When building RL for VLMs, teams repeatedly rebuild the same plumbing:
1. choose trainer backend (`trl` / `verl`)
2. wire train/reference model roles
3. apply vision-token masking correctly in loss
4. measure speed and memory in a repeatable way
5. iterate reward design quickly before expensive RL runs

AeroRL standardizes that plumbing so you can focus on the experiment itself.

## How it works (easy flow)

1. `wrap_vlm_for_rl(...)` prepares runtime metadata.
2. `AeroRLTrainer` runs `on_train_start()` → `train_step(...)` → `on_train_end()`.
3. `train_step(...)` uses masked cross-entropy (vision tokens excluded from text loss).
4. `vlm_grpo_benchmark.py` outputs JSON reports with `iters_per_sec` and `peak_vram_gb`.
5. `reward_replay_evaluator.py` scores offline trajectories with weighted reward components.

## 2-minute example (end-to-end)

```python
import torch
from aerorl import AeroRLConfig, AeroRLTrainer, wrap_vlm_for_rl

# 1) Build runtime config
cfg = AeroRLConfig(trainer_backend="auto", quant_ref_bits=8, mask_vision_tokens=True)
train_runtime, ref_runtime = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", cfg)
print("Trainer backend:", train_runtime["trainer"])
print("Reference mode:", ref_runtime["quantization_mode"])

# 2) One training step
trainer = AeroRLTrainer(cfg)
trainer.on_train_start()

logits = torch.randn(2, 4, 8, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
vision_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])

step_result = trainer.train_step(logits=logits, labels=labels, vision_mask=vision_mask)
print("Step result:", step_result)
print("Train end:", trainer.on_train_end())
```

## Benchmark command

```bash
python benchmarks/vlm_grpo_benchmark.py --mode real --model Qwen/Qwen2.5-VL-7B-Instruct --steps 20 --matrix-size 512
```

## Reward replay evaluator command

```bash
python benchmarks/reward_replay_evaluator.py --input reports/reward-replay-sample-2026-03-23.jsonl --output reports/reward-eval-summary-2026-03-23.json
```

## Measured results (current repo artifacts)

From `reports/benchmark-real-2026-03-23.json`:
- device: `cuda`
- throughput: `146.5458 it/s`
- peak VRAM: `0.0094 GB`

From `reports/benchmark-matrix-real-2026-03-23.json`:
- models benchmarked: `3`
- max peak VRAM: `0.0314 GB`

Validation status:
- test suite: `12 passed`

Reward evaluator artifact:
- `reports/reward-eval-summary-2026-03-23.json` (`average_reward`: `0.359` on sample replay set)

## Install

```bash
git clone https://github.com/sabdulmajid/aeroRL.git
cd aeroRL
python -m pip install -e .
```

## Typical use cases

- You are prototyping a new VLM-RL reward/loss idea and need clean trainer scaffolding.
- You need consistent benchmark JSONs for experiment tracking.
- You want a compact baseline before integrating a larger training stack.

## Public API at a glance

- `AeroRLConfig`
- `wrap_vlm_for_rl(...)`
- `AeroRLTrainer`
- `masked_cross_entropy_loss(...)`
- `create_quantized_reference_runtime(...)`
- `build_default_reward_stack(...)`
- `evaluate_records(...)`
