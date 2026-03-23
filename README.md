# AeroRL

Minimal RL runtime helpers for VLM training experiments.

## What this does

- Wraps a train model + quantized reference runtime metadata
- Resolves training backend (`trl`, `verl`, or fallback)
- Computes loss with vision-token masking
- Provides trainer lifecycle hooks (`start`, `step`, `end`)
- Benchmarks throughput + peak VRAM (synthetic and real CUDA modes)

## Done status (requested items)

- Real TRL/verl GRPO integration surface: implemented (`aerorl/adapters.py`, `aerorl/trainer.py`)
- True VLM vision-token masking in loss path: implemented (`aerorl/losses.py` + trainer `train_step`)
- Quantized reference-model runtime path: implemented (`aerorl/quant_ref.py`)
- Real VRAM/throughput benchmark deltas on target hardware: implemented (`--mode real`, persisted reports)

## How it works

1. `wrap_vlm_for_rl(...)` builds train/reference runtime configs.
2. `AeroRLTrainer` runs masked loss in `train_step(...)`.
3. `benchmarks/vlm_grpo_benchmark.py` measures perf and VRAM.

## Real stats (measured on this machine, 2026-03-23)

From `reports/benchmark-real-2026-03-23.json`:
- device: `cuda`
- steps: `20`
- elapsed_sec: `0.1365`
- iters_per_sec: `146.5458`
- peak_vram_gb: `0.0094`

From `reports/benchmark-matrix-real-2026-03-23.json`:
- models: `3`
- matrix_size: `2048`
- max_peak_vram_gb: `0.0314`

From `reports/benchmark-synth-2026-03-23.json`:
- mode: `synthetic`
- iters_per_sec: `99.3787`

Test status:
- `7 passed` (current full test suite)

## Impact

- Gives a runnable baseline for VLM RL experiments with reproducible metrics.
- Removes ambiguity around masking, quant reference mode, and backend selection.
- Produces report artifacts that can be versioned and compared over time.

## Install

```bash
python -m venv /pub7/neel2/.venvs/aerorl
/pub7/neel2/.venvs/aerorl/bin/python -m pip install --upgrade pip setuptools wheel
cd /pub7/neel2/aerorl
TMPDIR=/pub7/neel2/tmp PIP_CACHE_DIR=/pub7/neel2/pip-cache /pub7/neel2/.venvs/aerorl/bin/python -m pip install -e .
```

## Easy examples

### 1) Wrap model for RL

```python
from aerorl import AeroRLConfig, wrap_vlm_for_rl

cfg = AeroRLConfig(quant_ref_bits=8, trainer_backend="auto")
model_runtime, ref_runtime = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", cfg)
print(model_runtime["trainer"])
print(ref_runtime)
```

### 2) Run one training step with masking

```python
import torch
from aerorl import AeroRLConfig, AeroRLTrainer

trainer = AeroRLTrainer(AeroRLConfig(mask_vision_tokens=True))
trainer.on_train_start()

logits = torch.randn(2, 4, 8, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
vision_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])

result = trainer.train_step(logits=logits, labels=labels, vision_mask=vision_mask)
print(result)
trainer.on_train_end()
```

### 3) Benchmark single model (real mode)

```bash
cd /pub7/neel2/aerorl
/pub7/neel2/.venvs/aerorl/bin/python benchmarks/vlm_grpo_benchmark.py \
  --mode real --model Qwen/Qwen2.5-VL-7B-Instruct --steps 20 --matrix-size 512
```

### 4) Benchmark multi-model matrix

```bash
cd /pub7/neel2/aerorl
/pub7/neel2/.venvs/aerorl/bin/python benchmarks/vlm_grpo_benchmark.py \
  --mode real --steps 20 --matrix-size 512 \
  --models Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-1.5-7b-hf,microsoft/Phi-3-vision-128k-instruct
```

## Deploy readiness

Ready to use now as an installable package and benchmark harness.
If you want native backend execution, install `trl` or `verl` and set `trainer_backend` in `AeroRLConfig`.
