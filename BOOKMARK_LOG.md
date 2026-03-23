## 2026-03-23 — Session bootstrap (Copilot / GPT-5.3-Codex)

### What was done
- Reviewed existing repo files: `AGENTS.md`, `README.md`, `BOOKMARK_LOG.md`.
- Initialized a runnable Python package scaffold:
	- `pyproject.toml`
	- `aerorl/__init__.py`
	- `aerorl/config.py`
	- `aerorl/wrapper.py`
- Added synthetic benchmark entrypoint:
	- `benchmarks/vlm_grpo_benchmark.py`
- Added smoke test:
	- `tests/test_public_api.py`
- Updated `README.md` to match current implemented state.

### Why
- User requested durable progress that survives disconnects and can be resumed by another agent.
- Repo was docs-only and untracked; this creates a concrete baseline with executable artifacts.

### Repo state intent
- Keep public API stable via `aerorl.__init__` exports.
- Benchmark script is synthetic for smoke checks only; real benchmark integration is pending.

### Immediate next steps for next agent
1. Integrate a real VLM policy/ref-model loading path (likely HF + TRL/verl adapters).
2. Implement vision-mask-aware loss computation path and tests.
3. Add kernel scaffolding under `aerorl/kernels/` per AGENTS guidance.
4. Replace synthetic benchmark outputs with measured VRAM + throughput metrics.
5. Continue frequent small commits and push each checkpoint.

### Recovery instructions if session disconnects
- Run: `git -C /pub7/neel2/aerorl status`
- Read this file first, then `README.md`.
- Continue from "Immediate next steps" and commit after each logical sub-step.

### Environment notes from this session
- `python -m pip install -e .` failed with `OSError: [Errno 28] No space left on device`.
- `pytest` executable is not installed on host PATH.
- Functional smoke checks passed using `PYTHONPATH=. python ...` invocations.

## 2026-03-23 — Verification + disk-space fix

### What was done
- Diagnosed free space across mounted disks and confirmed `/pub7` has large free capacity.
- Created dedicated execution environment on `/pub7`:
	- venv: `/pub7/neel2/.venvs/aerorl`
	- temp: `/pub7/neel2/tmp`
	- pip cache: `/pub7/neel2/pip-cache`
- Successfully installed project with:
	- `TMPDIR=/pub7/neel2/tmp PIP_CACHE_DIR=/pub7/neel2/pip-cache /pub7/neel2/.venvs/aerorl/bin/python -m pip install -e .`
- Ran tests:
	- `/pub7/neel2/.venvs/aerorl/bin/python -m pytest -q`
	- Result: `1 passed in 0.01s`
- Ran benchmark and persisted output:
	- `/pub7/neel2/.venvs/aerorl/bin/python benchmarks/vlm_grpo_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct --steps 25`
	- Output saved to `reports/benchmark-smoke-2026-03-23.json`
- Added verification report:
	- `reports/verification-2026-03-23.md`

### Latest benchmark values
- `steps`: 25
- `elapsed_sec`: 0.2515
- `iters_per_sec`: 99.3925

### Resume note
- Use `/pub7/neel2/.venvs/aerorl/bin/python` for all project commands to avoid `/pub3` disk-space issues.

## 2026-03-23 — Completion pass for missing features

### User-required missing items addressed
- Real TRL/verl GRPO integration surface:
	- Added backend adapter resolution in `aerorl/adapters.py` and wired to `wrap_vlm_for_rl` output.
- Vision-token masking in loss path:
	- Added `aerorl/losses.py` with `build_text_token_mask` and `masked_cross_entropy_loss`.
- Quantized reference runtime path:
	- Added `aerorl/quant_ref.py` and integrated into wrapper outputs (`int4/int8/fp16-reference` modes).
- Real VRAM/throughput benchmark path:
	- Upgraded benchmark script to `--mode real` with measured `peak_vram_gb` and throughput.

### Validation summary
- `pytest`: `5 passed in 1.15s`
- Real benchmark (`cuda`) persisted to `reports/benchmark-real-2026-03-23.json`
- Synthetic benchmark persisted to `reports/benchmark-synth-2026-03-23.json`

### Environment constraints
- `trl` and `verl` are not installed in this machine, so backend auto-resolution reports scaffold mode.
- The integration layer is implemented and ready; full trainer run requires installing one of those backends.

## 2026-03-23 — Final simplify + done status pass

### Completed now
- Added `AeroRLTrainer` lifecycle + `train_step` integration for masked loss path.
- Added backend-aware quantization runtime selection (`torch` / `bitsandbytes` / `torchao` fallback logic).
- Added multi-model matrix benchmark mode with `--models`.
- Rewrote README to minimal user-facing format with direct stats and easy examples.

### Current validated state
- Tests: `7 passed`
- Real benchmark file: `reports/benchmark-real-2026-03-23.json`
- Real matrix file: `reports/benchmark-matrix-real-2026-03-23.json`

## 2026-03-23 — Reward stack + replay evaluator shipped

### Implemented
- Added `aerorl/rewards.py` with composable reward framework:
	- `VerifierReward`
	- `GroundingReward`
	- `FormatReward`
	- `CostReward`
	- `WeightedRewardStack`
	- `build_default_reward_stack`
	- `evaluate_records`
- Added offline evaluator CLI:
	- `benchmarks/reward_replay_evaluator.py`
- Exported reward APIs in `aerorl/__init__.py`.

### Tests and artifacts
- Added tests:
	- `tests/test_rewards.py`
	- `tests/test_reward_replay_evaluator_cli.py`
- Full suite result: `12 passed`.
- Added replay sample + summary artifacts:
	- `reports/reward-replay-sample-2026-03-23.jsonl`
	- `reports/reward-eval-summary-2026-03-23.json`

### Why this matters
- AeroRL now supports fast reward-iteration loops offline before expensive RL runs.
- This directly improves practical usefulness for reward-function experimentation.

## 2026-03-23 — Reward evaluator usability pass

### Improvements shipped
- `benchmarks/reward_replay_evaluator.py` now supports:
	- configurable weights via repeated `--weight name=value`
	- JSON/regex format constraints (`--require-json`, `--regex-pattern`)
	- cost controls (`--target-tokens`, `--latency-budget-ms`)
	- aggregate pass metrics (`--pass-threshold`)
	- best/worst surfacing (`--top-k`)
- `aerorl/rewards.py` now returns richer summaries:
	- `pass_rate`
	- `component_averages`
	- `best_examples` / `worst_examples`
- Added example replay dataset and output:
	- `examples/reward_replay_example.jsonl`
	- `reports/reward-eval-example-2026-03-23.json`

### Validation
- Full suite: `14 passed`
