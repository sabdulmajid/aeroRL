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

## Why this over doing it manually

Manual reward scoring usually breaks in three places: inconsistent formulas, no shared artifact format, and no fast way to debug failures.

AeroRL fixes that by giving you:
- one weighted reward contract (`verifier`, `grounding`, `format`, `cost`)
- one command to score full replay datasets
- one JSON output with aggregate metrics + best/worst examples for debugging

## Large-scale real benchmark: Manual vs AeroRL

Run:

```bash
python benchmarks/reward_real_dataset_benchmark.py \
	--report-output reports/reward-value-benchmark-real-large-2026-03-23.json \
	--replay-output reports/reward-replay-real-large-2026-03-23.jsonl
```

What this benchmark compares:
- **Manual baseline**: pass if `reference` appears in `response`
- **AeroRL stack**: weighted verifier + grounding + format + cost quality gate

Dataset source:
- `soarm100_cloth_fold_v1_clean.jsonl` + `soarm100_cloth_fold_v0.jsonl` manifests
- total episodes evaluated: **500**

Measured result (`reports/reward-value-benchmark-real-large-2026-03-23.json`):

| Metric | Manual | AeroRL | Why it matters |
|---|---:|---:|---|
| Dataset size | 500 | 500 | Large enough to avoid toy conclusions |
| Pass rate | 0.746 | 0.432 | AeroRL is a stricter quality gate |
| Quality dimensions checked | 1 | 4 | More complete quality signal |
| False passes caught | 0 | 161 | Manual accepted these; AeroRL flagged them |
| False pass rate among manual passes | N/A | 0.679325 caught | 67.93% of manual passes were hidden failures |

Component-level issues identified by AeroRL:
- format issues: `56`
- grounding issues: `86`
- verifier issues: `284`
- cost issues: `12`

Hidden failures caught by AeroRL:
- examples include episodes where manual label-matching passed but AeroRL detected:
	- invalid format
	- grounding mismatch
	- wrong verifier outcome

Bottom line:
- Manual scoring looked better on pass-rate only.
- AeroRL exposed large volumes of bad outputs that manual scoring would have accepted.
- This is the practical gain: **fewer false positives and clearer failure diagnosis**.

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

## Drop-in usage (copy/paste)

### 1) Dataset format (`.jsonl`)

Each line should be one record:

```json
{"id":"ex1","prompt":"...","response":"...","reference":"...","metadata":{"evidence_entities":["..."],"claimed_entities":["..."],"latency_ms":120}}
```

Use the included working example dataset:
- `examples/reward_replay_example.jsonl`

### 2) Run evaluator

```bash
python benchmarks/reward_replay_evaluator.py \
	--input examples/reward_replay_example.jsonl \
	--output reports/reward-eval-example-2026-03-23.json \
	--require-json \
	--regex-pattern '^\{.*\}$' \
	--weight verifier=0.45 \
	--weight grounding=0.3 \
	--weight format=0.2 \
	--weight cost=0.05 \
	--pass-threshold 0.5 \
	--top-k 2
```

### 3) Expected output (what you get)

Console summary includes:
- `count`
- `average_reward`
- `pass_rate`
- `component_averages`

JSON output includes:
- global metrics (`average_reward`, `pass_rate`, `weights`)
- `best_examples` and `worst_examples`
- per-record component scores and details

From the included example run (`reports/reward-eval-example-2026-03-23.json`):
- `average_reward`: `0.216333`
- `pass_rate`: `0.333333`
- best example: `good-ocr` (`total_reward=1.0`)
- worst example: `bad-grounding` (`total_reward=-0.275`)

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
- `reports/reward-eval-example-2026-03-23.json` (`average_reward`: `0.216333`, with best/worst examples)

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
