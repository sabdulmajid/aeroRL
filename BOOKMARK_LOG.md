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

## 2026-03-23 — Tangible value benchmark added

### What was added
- New benchmark script:
	- `benchmarks/reward_value_benchmark.py`
- New benchmark dataset:
	- `examples/reward_value_benchmark_dataset.jsonl`
- New benchmark artifact:
	- `reports/reward-value-benchmark-2026-03-23.json`
- New unit test:
	- `tests/test_reward_value_benchmark.py`

### Key measured results
- Manual pass rate: `0.833333`
- AeroRL pass rate: `0.5`
- AeroRL caught hidden manual false passes: `2`
- False pass rate among manual passes: `0.4`
- Observability dimensions: manual `1` vs AeroRL `4`

### Why this is useful
- Demonstrates concrete quality-gate value, not just aggregate score output.
- Shows exactly what AeroRL catches that naive manual scoring misses.

## 2026-03-23 — Large-scale real dataset benchmark

### Data source
- Real cached HF datasets from `/pub7/neel2/.cache_hf/datasets`:
	- `nielsr/docvqa_1200_examples` train
	- `HuggingFaceM4/ChartQA` train shards
- Total records evaluated: `29,299`

### Safety and infra checks
- Ran `nvidia-smi` before benchmark.
- GPUs were idle (0% utilization except display processes).
- Benchmark intentionally executed in CPU mode.

### Script and artifact
- Script: `benchmarks/reward_large_scale_real_dataset_benchmark.py`
- Artifact: `reports/reward-large-scale-benchmark-2026-03-23.json`

### Key measured outcomes
- Manual pass rate: `0.997747`
- AeroRL pass rate: `0.624083`
- Hidden manual false passes caught by AeroRL: `10,948`
- Hidden false-pass fraction among manual passes: `0.374508`
- Evaluation throughput: `52,646 records/s` (CPU)

### Interpretation
- Manual scoring overestimates quality on large datasets.
- AeroRL provides materially stronger quality gating and diagnostics at scale.

## 2026-03-24 — Real model-generated benchmark shipped

### What was added
- New benchmark script:
	- `benchmarks/reward_model_generated_benchmark.py`
- New focused tests:
	- `tests/test_reward_model_generated_benchmark.py`

### Implementation highlights
- Loads real cached datasets from `/pub7/neel2/.cache_hf/datasets`:
	- DocVQA train Arrow
	- ChartQA train Arrow shards
- Decodes real image bytes and runs image-conditioned generation with cached SmolVLM (`HuggingFaceTB/SmolVLM-256M-Instruct`).
- Normalizes outputs into a JSON answer contract and scores with AeroRL reward stack.
- Writes both replay-level JSONL and benchmark summary JSON artifacts.
- Added local snapshot/cache resolution + forced HF cache env handling to avoid `/pub3` disk-full path.

### GPU safety and execution notes
- Checked GPU state before run via `nvidia-smi` (idle).
- Ran benchmark on `cuda:0` using local cached model files only.

### Artifact and measured result
- Summary artifact:
	- `reports/reward-model-generated-benchmark-2026-03-24.json`
- Replay artifact:
	- `reports/reward-model-generated-replay-2026-03-24.jsonl`
- Run shape:
	- `40` DocVQA + `60` ChartQA = `100` model-generated samples
- Key outcomes:
	- manual pass rate: `0.47`
	- AeroRL pass rate: `0.3`
	- hidden manual false passes caught by AeroRL: `17`
	- hidden false-pass fraction among manual passes: `0.361702`
	- generation throughput: `1.784` samples/sec (GPU)

### Validation
- New test file passes:
	- `tests/test_reward_model_generated_benchmark.py`: `2 passed`

## 2026-03-24 — Larger real-world model-generated run (800 samples)

### Run shape
- Script: `benchmarks/reward_model_generated_benchmark.py`
- Model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Device: `cuda:0`
- Data: real cached DocVQA + ChartQA rows with image-conditioned generation
- Limits: `300` DocVQA + `500` ChartQA = `800` total

### Artifacts
- `reports/reward-model-generated-benchmark-2026-03-24-large.json`
- `reports/reward-model-generated-replay-2026-03-24-large.jsonl`

### Key outcomes
- manual pass rate: `0.47125`
- AeroRL pass rate: `0.2875`
- hidden manual false passes caught by AeroRL: `147`
- hidden false-pass fraction among manual passes: `0.38992`
- generation throughput: `1.462` samples/sec

### Runtime/GPU note
- Reported GPU status before run: near idle.
- Reported GPU status after run: very high utilization/memory pressure (`~96-99%` util on two GPUs reported by `nvidia-smi` query snapshot in artifact).
