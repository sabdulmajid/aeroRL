# AeroRL

AeroRL is a compact library for vision-language RL experimentation. It packages the repetitive parts of VLM-RL work that teams usually rebuild by hand: trainer/runtime wiring, masked loss over mixed vision-language tokens, lightweight benchmarking, and replayable reward evaluation.

## What AeroRL Covers

- runtime setup for train and reference model roles
- masked cross-entropy for text tokens with vision-token exclusion
- a minimal trainer lifecycle for experimentation
- a weighted reward stack for offline replay scoring
- benchmark scripts for throughput, VRAM, and reward-quality checks

## Why It Exists

MLE teams working on VLM-RL usually need the same scaffolding before they can even test an idea:

1. choose and normalize trainer backend behavior
2. wire train/reference model roles
3. get token masking right for mixed-modal sequences
4. measure throughput and memory in a repeatable way
5. iterate on reward logic before spending GPU time on RL

AeroRL standardizes that layer so reward design and experiment iteration are easier to reason about and easier to reproduce.

## Install

Core library:

```bash
git clone https://github.com/sabdulmajid/aeroRL.git
cd aeroRL
python -m pip install -e .
```

Library + benchmark tooling + test runner:

```bash
python -m pip install -e ".[benchmark,dev]"
```

`benchmark` installs the image / Arrow / Transformers dependencies used by the benchmark scripts. `dev` installs `pytest`.

Library + local support for the external `lmms-eval` benchmark wrapper:

```bash
python -m pip install -e ".[benchmark,benchmark_standard,dev]"
python -m pip install --no-deps git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@v0.7.1
```

The extra step is intentional. `lmms-eval` has a very wide dependency surface, so AeroRL keeps the CUDA / PyTorch stack under your control and installs the harness itself without letting it re-resolve `torch`.

## 2-Minute Example

```python
import torch
from aerorl import AeroRLConfig, AeroRLTrainer, wrap_vlm_for_rl

cfg = AeroRLConfig(trainer_backend="auto", quant_ref_bits=8, mask_vision_tokens=True)
train_runtime, ref_runtime = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", cfg)

trainer = AeroRLTrainer(cfg)
trainer.on_train_start()

logits = torch.randn(2, 4, 8, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
vision_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])

step_result = trainer.train_step(logits=logits, labels=labels, vision_mask=vision_mask)

print(train_runtime["trainer"])
print(ref_runtime["quantization_mode"])
print(step_result)
print(trainer.on_train_end())
```

## Reward Replay Contract

The replay evaluator expects newline-delimited JSON records with `prompt`, `response`, `reference`, and `metadata`.

`reference` may be either:

- a single canonical answer string
- a list of acceptable answer strings when the dataset provides multiple valid aliases

Example:

```json
{"id":"ex1","prompt":"...","response":"{\"answer\":\"april 15, 2014\"}","reference":["april 15,2014","april 15, 2014"],"metadata":{"evidence_entities":["april 15,2014","april 15, 2014"],"claimed_entities":["april 15, 2014"],"latency_ms":120}}
```

The built-in reward stack combines four components:

- `verifier`: answer correctness against one or more acceptable references
- `grounding`: claimed vs evidence entity consistency with normalization for common surface-form variants
- `format`: JSON / regex contract checks
- `cost`: latency and verbosity pressure

Run the replay evaluator:

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

## Benchmarks

### Reward Value Sanity Check

Small deterministic benchmark:

```bash
python benchmarks/reward_value_benchmark.py \
  --input examples/reward_value_benchmark_dataset.jsonl \
  --output reports/reward-value-benchmark-2026-03-23.json
```

Current checked-in result:

- dataset size: `6`
- manual pass rate: `0.833333`
- AeroRL pass rate: `0.5`
- hidden false passes caught by AeroRL: `2`

This benchmark is intentionally tiny. Its job is to show the scoring contract, not to claim model quality.

### Large Cached Replay Benchmark

CPU-only replay benchmark over cached DocVQA + ChartQA rows:

```bash
python benchmarks/reward_large_scale_real_dataset_benchmark.py \
  --limit 50000 \
  --report-output reports/reward-large-scale-benchmark-2026-03-23.json
```

Current checked-in artifact: `reports/reward-large-scale-benchmark-2026-03-23.json`

- total records: `29,299`
- manual pass rate: `0.997747`
- AeroRL pass rate: `0.624083`
- hidden false passes caught: `10,948`
- evaluation throughput: `52,646` records/s on CPU

This benchmark uses deterministic corruption patterns over real prompts and references. It is useful for stress-testing the reward stack, but it is not a substitute for model-generated evaluation.

### Model-Generated Benchmark

This is the primary benchmark in the repo. It uses real cached images and questions, generates answers with a real VLM, and then scores those answers with both the manual baseline and the AeroRL reward stack.

First-principles takeaway:

- the old March 27 small-model run underused the GPU because it paired a 256M model with batch size `1`
- the current benchmark fixes that by batching generation and sampling GPU usage continuously during the run
- the goal is not to maximize VRAM for its own sake; the goal is to maximize useful throughput while keeping quality strong

Always preflight GPU usage before a run:

```bash
nvidia-smi
```

### Model Matrix

We first ran a balanced 200-sample matrix on one physical GPU to find a good model / batch-size point:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/reward_model_generated_benchmark.py \
  --models 'HuggingFaceTB/SmolVLM-500M-Instruct@32,Qwen/Qwen2-VL-2B-Instruct@16,Qwen/Qwen2.5-VL-3B-Instruct@16,Qwen/Qwen2.5-VL-7B-Instruct@8' \
  --device cuda:0 \
  --cache-dir /pub7/neel2/.cache_hf \
  --limit-docvqa 100 \
  --limit-chartqa 100 \
  --max-new-tokens 16 \
  --gpu-sample-interval-sec 0.5 \
  --output reports/reward-model-generated-matrix-2026-03-27.json
```

Artifact: `reports/reward-model-generated-matrix-2026-03-27.json`

| Model | Batch | Samples/s | AeroRL pass rate | Avg GPU util | Peak reserved GB |
|---|---:|---:|---:|---:|---:|
| `SmolVLM-500M` | 32 | 0.835 | 0.69 | 2.783% | 18.459 |
| `Qwen2-VL-2B` | 16 | 2.416 | 0.76 | 84.947% | 48.008 |
| `Qwen2.5-VL-3B` | 16 | 1.941 | 0.61 | 73.409% | 43.316 |
| `Qwen2.5-VL-7B` | 8 | 1.553 | 0.39 | 64.632% | 42.719 |

What this means:

- On this machine, `Qwen2-VL-2B-Instruct` was the best measured balance of quality, throughput, and actual GPU use.
- Bigger was not automatically better. The 7B model used more compute than the old tiny baseline, but it did not beat the 2B model on this slice.
- Batching was the real lever. That is what moved the GPU from “barely used” to “meaningfully busy.”

### Headline 800-Sample Run

After the matrix, we promoted the winning configuration to the full dated run:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/reward_model_generated_benchmark.py \
  --model-id 'Qwen/Qwen2-VL-2B-Instruct' \
  --device cuda:0 \
  --cache-dir /pub7/neel2/.cache_hf \
  --limit-docvqa 300 \
  --limit-chartqa 500 \
  --batch-size 16 \
  --max-new-tokens 16 \
  --gpu-sample-interval-sec 0.5 \
  --output reports/reward-model-generated-benchmark-qwen2-vl-2b-2026-03-27.json \
  --replay-output reports/reward-model-generated-replay-qwen2-vl-2b-2026-03-27.jsonl
```

Artifact: `reports/reward-model-generated-benchmark-qwen2-vl-2b-2026-03-27.json`

- model: `Qwen/Qwen2-VL-2B-Instruct`
- process-visible device: `cuda:0`
- physical GPU used: `GPU 1` via `CUDA_VISIBLE_DEVICES=1`
- total samples: `800`
- batch size: `16`
- generation throughput: `3.05` samples/s
- token throughput: `13.13` generated tokens/s
- average sample latency: `327.795 ms`
- p95 sample latency: `857.34 ms`
- average GPU utilization during the run: `86.845%`
- max GPU utilization during the run: `100%`
- average GPU memory in use during the run: `57,742.51 MiB`
- max GPU memory in use during the run: `68,702 MiB`
- peak PyTorch reserved memory: `66.404 GB`
- manual pass rate: `0.71625`
- AeroRL pass rate: `0.6825`
- manual false passes caught by AeroRL: `27`
- replay rows with multiple acceptable references preserved: `122`
- empty generations recorded as prompt echoes: `0`

The model-generated benchmark is also where the recent scoring fixes matter most:

- DocVQA rows now keep every acceptable answer alias instead of collapsing to one string.
- Grounding tolerates common surface-form variants like trailing punctuation and percent formatting.
- Empty generations stay empty instead of being decoded as the prompt template.

### Public Harness Benchmark

This benchmark answers a different question than the AeroRL reward benchmark:

- the AeroRL benchmark measures reward quality and replay integrity on real model generations
- the public harness benchmark measures the same model on a standard external evaluation stack

We keep the external harness at batch size `1` on purpose. `lmms-eval` explicitly warns that some multimodal backends can change behavior across batch sizes, so for the README numbers below we chose the conservative setting over maximum GPU saturation.

Run the public lite VQA suite:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/lmms_eval_standard_benchmark.py \
  --device cuda:0 \
  --batch-size 1 \
  --process-with-media \
  --force-simple \
  --output reports/lmms-eval-standard-benchmark-2026-03-27.json \
  --raw-output-dir reports/lmms-eval-standard-raw-2026-03-27 \
  --log-output reports/lmms-eval-standard-2026-03-27.log
```

Artifact: `reports/lmms-eval-standard-benchmark-2026-03-27.json`

Public-harness results for `Qwen/Qwen2-VL-2B-Instruct`:

| Task | Samples | Metric | Score |
|---|---:|---|---:|
| `aerorl_docvqa_val_lite` | 500 | `anls` | 0.860572 |
| `aerorl_chartqa_lite` | 500 | `relaxed_overall` | 0.650000 |

Run summary:

- total public-harness samples: `1,000`
- elapsed time: `235.42 s`
- throughput: `4.248` samples/s
- physical GPU used: `GPU 1` via `CUDA_VISIBLE_DEVICES=1`
- average GPU utilization during the run: `27.249%`
- max GPU utilization during the run: `61%`
- average GPU memory in use during the run: `4,769.815 MiB`
- max GPU memory in use during the run: `5,230 MiB`
- average GPU power draw during the run: `157.478 W`
- max GPU power draw during the run: `213.42 W`

What this means:

- AeroRL now has a public benchmark path that does not depend on AeroRL’s own scoring logic.
- The external harness run is intentionally easier to compare across teams, but it is not the right benchmark to maximize single-GPU utilization.
- The batched AeroRL benchmark remains the right place to measure local throughput and reward-pipeline behavior.

## Synthetic Smoke Benchmark

`benchmarks/vlm_grpo_benchmark.py` is still useful, but it should be read as a synthetic torch smoke benchmark, not the primary README performance claim. The README performance story above comes from real model generation on real image/question pairs.

```bash
python benchmarks/vlm_grpo_benchmark.py \
  --mode real \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --steps 20 \
  --matrix-size 512
```

Current checked-in artifacts:

- `reports/benchmark-real-2026-03-23.json`
- `reports/benchmark-matrix-real-2026-03-23.json`

## Public API

- `AeroRLConfig`
- `wrap_vlm_for_rl(...)`
- `AeroRLTrainer`
- `masked_cross_entropy_loss(...)`
- `create_quantized_reference_runtime(...)`
- `build_default_reward_stack(...)`
- `build_reward_stack(...)`
- `evaluate_records(...)`

## Validation

Local validation completed on March 27, 2026:

- `python -m pytest -q`
- result: `29 passed`
- model-generated matrix: `reports/reward-model-generated-matrix-2026-03-27.json`
- model-generated dated benchmark: `reports/reward-model-generated-benchmark-qwen2-vl-2b-2026-03-27.json`
- public harness benchmark: `reports/lmms-eval-standard-benchmark-2026-03-27.json`

## Typical Use Cases

- prototyping VLM-RL reward or loss ideas before integrating into a larger trainer
- replay-scoring offline trajectories to debug false positives and hidden failures
- generating dated JSON artifacts for throughput, VRAM, and reward-quality tracking
