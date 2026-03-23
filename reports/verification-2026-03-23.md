# AeroRL Verification Report — 2026-03-23

## Environment workaround
- Root cause: original active Python environment lived on a full disk (`/pub3`), causing install failure.
- Resolution: created dedicated environment on `/pub7`:
  - venv: `/pub7/neel2/.venvs/aerorl`
  - temp dir: `/pub7/neel2/tmp`
  - pip cache: `/pub7/neel2/pip-cache`

## Commands executed

```bash
# environment bootstrap
python -m venv /pub7/neel2/.venvs/aerorl
/pub7/neel2/.venvs/aerorl/bin/python -m pip install --upgrade pip setuptools wheel pytest

# project install
cd /pub7/neel2/aerorl
TMPDIR=/pub7/neel2/tmp PIP_CACHE_DIR=/pub7/neel2/pip-cache /pub7/neel2/.venvs/aerorl/bin/python -m pip install -e .

# tests
/pub7/neel2/.venvs/aerorl/bin/python -m pytest -q

# benchmark
/pub7/neel2/.venvs/aerorl/bin/python benchmarks/vlm_grpo_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct --steps 25
```

## Results

### Test results
- `1 passed in 0.01s`

### Benchmark results
- Report file: `reports/benchmark-smoke-2026-03-23.json`
- Key values:
  - `steps`: 25
  - `elapsed_sec`: 0.2515
  - `iters_per_sec`: 99.3925
  - `model`: `Qwen/Qwen2.5-VL-7B-Instruct`

## Status
- Baseline scaffold is installed, validated, and reproducible with explicit commands.
- Current benchmark is synthetic smoke coverage; real VLM GRPO measurement integration remains future scope.
