# AeroRL

Zero-copy-ready RL scaffolding for Vision-Language Models.

This repository is now initialized with a minimal, runnable baseline so work can continue safely across sessions and handoffs.

## Current Status

- Python package scaffold is in place (`aerorl/`)
- Public API exports:
	- `AeroRLConfig`
	- `wrap_vlm_for_rl(...)`
- Synthetic benchmark entrypoint available in `benchmarks/vlm_grpo_benchmark.py`
- Smoke test in `tests/test_public_api.py`

## Quick Start (Dev)

```bash
python -m pip install -e .
python benchmarks/vlm_grpo_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct --steps 10
pytest -q
```

## Python API

```python
from aerorl import AeroRLConfig, wrap_vlm_for_rl

config = AeroRLConfig(zero_copy_kv=True, mask_vision_tokens=True, quant_ref_bits=8)
model, ref_model = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", config)
```

## Next Build Targets

1. Replace synthetic benchmark with real GRPO/verl-compatible benchmark runner
2. Add true vision-token masking into loss paths
3. Implement quantized reference model runtime path
4. Add kernel package in `aerorl/kernels/` and profile on RTX PRO 6000 96 GB
