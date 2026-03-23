# AeroRL

Memory-efficient RL infrastructure for Vision-Language Models (VLMs), with a focus on reproducibility, clear API boundaries, and benchmark-driven development.

## Project Status

This repository is currently at **scaffold + verified smoke baseline** stage.

Implemented now:
- Package layout and installable Python project
- Stable public API surface:
	- `AeroRLConfig`
	- `wrap_vlm_for_rl(...)`
- Synthetic benchmark runner for execution-path validation
- Smoke test coverage for public API contract
- Session handoff logging to survive disconnects and agent swaps

Not implemented yet:
- Real TRL/verl GRPO training integration
- True VLM vision-token masking in loss kernels
- Quantized reference-model runtime path
- Real VRAM/throughput benchmark deltas on target hardware

## Repository Structure

- `aerorl/` — package source
- `benchmarks/` — benchmark entrypoints
- `tests/` — unit/smoke tests
- `reports/` — persisted run and verification artifacts
- `BOOKMARK_LOG.md` — persistent execution and handoff log

## Reproducible Setup

Recommended (disk-safe path used during verification):

```bash
python -m venv /pub7/neel2/.venvs/aerorl
/pub7/neel2/.venvs/aerorl/bin/python -m pip install --upgrade pip setuptools wheel pytest

cd /pub7/neel2/aerorl
TMPDIR=/pub7/neel2/tmp PIP_CACHE_DIR=/pub7/neel2/pip-cache /pub7/neel2/.venvs/aerorl/bin/python -m pip install -e .
```

## Verification Commands

```bash
cd /pub7/neel2/aerorl
/pub7/neel2/.venvs/aerorl/bin/python -m pytest -q
/pub7/neel2/.venvs/aerorl/bin/python benchmarks/vlm_grpo_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct --steps 25
```

## Verified Results (2026-03-23)

- Tests: `1 passed in 0.01s`
- Benchmark artifact: `reports/benchmark-smoke-2026-03-23.json`
- Benchmark values:
	- `steps`: 25
	- `elapsed_sec`: 0.2515
	- `iters_per_sec`: 99.3925

Full run log: `reports/verification-2026-03-23.md`

## Public API Example

```python
from aerorl import AeroRLConfig, wrap_vlm_for_rl

config = AeroRLConfig(zero_copy_kv=True, mask_vision_tokens=True, quant_ref_bits=8)
model, ref_model = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", config)
```

## Engineering Standards

- Keep public API stable via `aerorl/__init__.py`
- Record all progress in `BOOKMARK_LOG.md`
- Commit small, frequent checkpoints and push continuously
- Prefer benchmark-backed claims over aspirational performance statements

## Roadmap (Execution Order)

1. Real model loading path (HF + TRL/verl adapter boundary)
2. Vision-token-mask-aware loss pipeline + tests
3. Quantized reference model path and validation
4. Kernel package under `aerorl/kernels/`
5. Hardware benchmark suite with VRAM + throughput comparisons

## Contribution Note

For handoff-friendly collaboration, start from `BOOKMARK_LOG.md`, run verification commands, then continue roadmap items with one commit per logical unit of work.
