from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

from aerorl import build_reward_stack, evaluate_records

DOCVQA_TRAIN = Path(
    "/pub7/neel2/.cache_hf/datasets/nielsr___docvqa_1200_examples/default/0.0.0/"
    "dc77ab0c3d98855d0f3cb3a00832f2423fbe7528/docvqa_1200_examples-train.arrow"
)

CHARTQA_TRAIN_FILES = [
    Path(
        "/pub7/neel2/.cache_hf/datasets/HuggingFaceM4___chart_qa/default/0.0.0/"
        "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5/chart_qa-train-00000-of-00003.arrow"
    ),
    Path(
        "/pub7/neel2/.cache_hf/datasets/HuggingFaceM4___chart_qa/default/0.0.0/"
        "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5/chart_qa-train-00001-of-00003.arrow"
    ),
    Path(
        "/pub7/neel2/.cache_hf/datasets/HuggingFaceM4___chart_qa/default/0.0.0/"
        "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5/chart_qa-train-00002-of-00003.arrow"
    ),
]


@dataclass(slots=True)
class Sample:
    sample_id: str
    dataset: str
    prompt: str
    references: tuple[str, ...]
    image: Image.Image


class GPUSampler:
    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = max(interval_sec, 0.1)
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self._capture_once()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec * 4)
            self._thread = None
        self._capture_once()
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._capture_once()

    def _capture_once(self) -> None:
        status = _gpu_status()
        if status.get("available"):
            self.samples.append(
                {
                    "timestamp_sec": round(time.time(), 3),
                    "gpus": [dict(gpu) for gpu in status.get("gpus", [])],
                }
            )


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _gpu_status() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    gpus: list[dict[str, Any]] = []
    for line in out:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "utilization_gpu_pct": int(fields[4]),
                "power_draw_w": _safe_float(fields[5]),
            }
        )

    return {"available": True, "gpus": gpus}


def _read_arrow_table(path: Path) -> pa.Table:
    mm = pa.memory_map(str(path), "r")
    try:
        return ipc.open_file(mm).read_all()
    except Exception:
        mm.seek(0)
        return ipc.open_stream(mm).read_all()


def _decode_image(image_payload: Any) -> Image.Image:
    if isinstance(image_payload, dict):
        raw = image_payload.get("bytes")
        path = image_payload.get("path")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(image_payload, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_payload)).convert("RGB")
    raise ValueError("unsupported image payload")


def _safe_text(value: Any) -> str:
    if isinstance(value, dict):
        if "en" in value and value["en"]:
            return str(value["en"])
        for item in value.values():
            if item:
                return str(item)
        return ""
    if isinstance(value, list):
        for item in value:
            if str(item).strip():
                return str(item)
        return ""
    return str(value)


def _safe_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_safe_text_list(item))
        seen: set[str] = set()
        unique_items: list[str] = []
        for item in items:
            if item not in seen:
                unique_items.append(item)
                seen.add(item)
        return unique_items

    text = _safe_text(value).strip()
    return [text] if text else []


def _reference_texts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _iter_sample_batches(samples: list[Sample], batch_size: int) -> Iterator[list[Sample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    avg = sum(values) / len(values)
    return {
        "avg": round(avg, 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def _resolve_visible_gpu_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    return int(device.split(":", 1)[1])


def _resolve_physical_gpu_index(device: str) -> int | None:
    visible_index = _resolve_visible_gpu_index(device)
    if visible_index is None:
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return visible_index

    visible_ids = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if visible_index >= len(visible_ids):
        return None
    if visible_ids[visible_index].isdigit():
        return int(visible_ids[visible_index])
    return None


def _summarize_gpu_samples(samples: list[dict[str, Any]], physical_gpu_index: int | None) -> dict[str, Any]:
    if physical_gpu_index is None:
        return {}

    selected: list[dict[str, Any]] = []
    for sample in samples:
        gpu = next((item for item in sample.get("gpus", []) if item.get("index") == physical_gpu_index), None)
        if gpu is not None:
            selected.append(gpu)

    if not selected:
        return {"physical_gpu_index": physical_gpu_index, "sample_count": 0}

    memory_values = [float(item["memory_used_mib"]) for item in selected]
    utilization_values = [float(item["utilization_gpu_pct"]) for item in selected]
    power_values = [float(item["power_draw_w"]) for item in selected if item.get("power_draw_w") is not None]

    profile = {
        "physical_gpu_index": physical_gpu_index,
        "sample_count": len(selected),
        "avg_memory_used_mib": round(sum(memory_values) / len(memory_values), 3),
        "max_memory_used_mib": round(max(memory_values), 3),
        "avg_utilization_gpu_pct": round(sum(utilization_values) / len(utilization_values), 3),
        "max_utilization_gpu_pct": round(max(utilization_values), 3),
    }
    if power_values:
        profile["avg_power_draw_w"] = round(sum(power_values) / len(power_values), 3)
        profile["max_power_draw_w"] = round(max(power_values), 3)
    return profile


def load_samples(limit_docvqa: int, limit_chartqa: int) -> list[Sample]:
    samples: list[Sample] = []

    if limit_docvqa > 0:
        doc = _read_arrow_table(DOCVQA_TRAIN)
        for row in doc.to_pylist()[:limit_docvqa]:
            prompt = _safe_text(row.get("query", "")).strip()
            references = _safe_text_list(row.get("answers", row.get("answer", "")))
            if not prompt or not references:
                continue
            samples.append(
                Sample(
                    sample_id=f"docvqa::{row.get('id', len(samples))}",
                    dataset="docvqa",
                    prompt=prompt,
                    references=tuple(references),
                    image=_decode_image(row.get("image")),
                )
            )

    if limit_chartqa > 0:
        remaining = limit_chartqa
        for shard_idx, shard_path in enumerate(CHARTQA_TRAIN_FILES):
            if remaining <= 0:
                break
            table = _read_arrow_table(shard_path)
            rows = table.to_pylist()
            for row_idx, row in enumerate(rows):
                if remaining <= 0:
                    break
                prompt = _safe_text(row.get("query", "")).strip()
                references = _safe_text_list(row.get("label", row.get("answer", "")))
                if not prompt or not references:
                    continue
                samples.append(
                    Sample(
                        sample_id=f"chartqa::{shard_idx}:{row_idx}",
                        dataset="chartqa",
                        prompt=prompt,
                        references=tuple(references),
                        image=_decode_image(row.get("image")),
                    )
                )
                remaining -= 1

    return samples


def _resolve_torch_dtype(config: Any, device: str) -> Any:
    import torch

    raw_dtype = getattr(config, "torch_dtype", None)
    if raw_dtype is None:
        return torch.float32 if not device.startswith("cuda") else torch.bfloat16
    if raw_dtype == "auto":
        return torch.float32 if not device.startswith("cuda") else torch.bfloat16
    if isinstance(raw_dtype, str):
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(raw_dtype, torch.float32 if not device.startswith("cuda") else torch.bfloat16)
    return raw_dtype


def _load_model(model_id: str, cache_dir: str | None, device: str):
    from transformers import AutoConfig, AutoProcessor

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

    resolved_model_id = _resolve_model_path(model_id=model_id, cache_dir=cache_dir)
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(cache_dir) / "hub")

    config = AutoConfig.from_pretrained(resolved_model_id, cache_dir=cache_dir, local_files_only=True)
    torch_dtype = _resolve_torch_dtype(config=config, device=device)
    processor = AutoProcessor.from_pretrained(resolved_model_id, cache_dir=cache_dir, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", "") != "left":
        tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        resolved_model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        dtype=torch_dtype,
    ).to(device)
    model.eval()
    return processor, model


def _resolve_model_path(model_id: str, cache_dir: str | None) -> str:
    if Path(model_id).exists():
        return model_id
    if cache_dir is None:
        return model_id

    repo_dir = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    snapshots = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").exists() else []
    if snapshots:
        return str(snapshots[-1])
    return model_id


def _extract_generated_ids(output_ids: Any, input_ids: Any | None) -> Any:
    if input_ids is not None:
        try:
            return output_ids[:, input_ids.shape[1] :]
        except Exception:
            return output_ids
    return output_ids


def _generated_token_counts(generated_ids: Any, pad_token_id: int | None, eos_token_id: int | None) -> list[int]:
    invalid_ids = {token_id for token_id in [pad_token_id, eos_token_id] if token_id is not None}
    counts: list[int] = []
    for row in generated_ids:
        row_ids = row.tolist() if hasattr(row, "tolist") else list(row)
        counts.append(sum(1 for token_id in row_ids if token_id not in invalid_ids))
    return counts


def _decode_generated_texts(processor: Any, output_ids: Any, input_ids: Any | None) -> tuple[list[str], list[int]]:
    generated_ids = _extract_generated_ids(output_ids, input_ids)
    batch_size = int(output_ids.shape[0]) if hasattr(output_ids, "shape") and len(output_ids.shape) > 0 else 1

    try:
        if generated_ids.shape[-1] == 0:
            return [""] * batch_size, [0] * batch_size
    except Exception:
        pass

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    texts = [item.strip() for item in decoded]

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    counts = _generated_token_counts(generated_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id)

    if len(texts) < batch_size:
        texts.extend([""] * (batch_size - len(texts)))
    if len(counts) < batch_size:
        counts.extend([0] * (batch_size - len(counts)))
    return texts, counts


def _decode_generated_text(processor: Any, output_ids: Any, input_ids: Any | None) -> str:
    texts, _ = _decode_generated_texts(processor, output_ids, input_ids)
    return texts[0].strip() if texts else ""


def _response_to_json_answer(response: str) -> tuple[str, str, bool]:
    text = response.strip()
    parsed_answer = text
    was_json = False

    try:
        payload = json.loads(text)
        was_json = isinstance(payload, dict)
        if isinstance(payload, dict) and "answer" in payload:
            parsed_answer = str(payload["answer"]).strip()
        elif isinstance(payload, dict):
            parsed_answer = _safe_text(list(payload.values())).strip()
    except json.JSONDecodeError:
        parsed_answer = text

    wrapped = json.dumps({"answer": parsed_answer}, ensure_ascii=False, separators=(",", ":"))
    return wrapped, parsed_answer, was_json


def _build_chat_prompt(question: str) -> str:
    return (
        "Answer the question from the image. "
        "Return strict JSON only with one key: answer. "
        f"Question: {question}"
    )


def _parse_model_specs(raw_specs: str, default_batch_size: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for raw_spec in raw_specs.split(","):
        item = raw_spec.strip()
        if not item:
            continue
        if "@" in item:
            model_id, raw_batch_size = item.rsplit("@", 1)
            batch_size = int(raw_batch_size)
        else:
            model_id = item
            batch_size = default_batch_size
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size in model spec '{item}'")
        specs.append({"model_id": model_id.strip(), "batch_size": batch_size})
    return specs


def manual_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = 0
    pass_ids: list[str] = []
    for row in records:
        response = str(row["response"]).lower()
        normalized_references = [reference.lower() for reference in _reference_texts(row.get("reference"))]
        ok = any(reference in response for reference in normalized_references)
        passed += int(ok)
        if ok:
            pass_ids.append(str(row["id"]))
    total = len(records)
    return {
        "method": "manual_contains_reference",
        "pass_rate": round((passed / total) if total else 0.0, 6),
        "pass_ids": pass_ids,
    }


def build_model_generated_report(
    *,
    records: list[dict[str, Any]],
    aerorl: dict[str, Any],
    manual: dict[str, Any],
    model_id: str,
    device: str,
    generate_elapsed_sec: float,
    load_elapsed_sec: float,
    gpu_before: dict[str, Any],
    gpu_after: dict[str, Any],
    prompt_style: str,
    max_new_tokens: int,
    batch_size: int = 1,
    total_generated_tokens: int = 0,
    batch_latency_ms: list[float] | None = None,
    sample_latency_ms: list[float] | None = None,
    gpu_profile: dict[str, Any] | None = None,
    torch_cuda_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manual_pass_ids = set(manual["pass_ids"])
    aerorl_fail_ids = {
        item["id"]
        for item in aerorl["results"]
        if item["reward"]["total_reward"] < aerorl["pass_threshold"]
    }
    false_passes_caught = sorted(manual_pass_ids.intersection(aerorl_fail_ids))

    doc_count = sum(1 for row in records if str(row.get("id", "")).startswith("docvqa::"))
    chart_count = sum(1 for row in records if str(row.get("id", "")).startswith("chartqa::"))
    batch_latency_summary = _summarize_values(batch_latency_ms or [])
    sample_latency_summary = _summarize_values(sample_latency_ms or [])

    return {
        "dataset": {
            "name": "DocVQA + ChartQA (real images, model-generated outputs)",
            "total_records": len(records),
            "sources": {
                "docvqa_rows": doc_count,
                "chartqa_rows": chart_count,
            },
            "prompt_style": prompt_style,
        },
        "generation": {
            "model_id": model_id,
            "device": device,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "load_sec": round(load_elapsed_sec, 3),
            "generate_sec": round(generate_elapsed_sec, 3),
            "samples_per_sec": round((len(records) / generate_elapsed_sec) if generate_elapsed_sec else 0.0, 3),
            "generated_tokens_total": total_generated_tokens,
            "generated_tokens_per_sec": round(
                (total_generated_tokens / generate_elapsed_sec) if generate_elapsed_sec else 0.0,
                3,
            ),
            "batch_latency_ms": batch_latency_summary,
            "sample_latency_ms": sample_latency_summary,
            "gpu_status_before": gpu_before,
            "gpu_status_after": gpu_after,
            "gpu_profile": gpu_profile or {},
            "torch_cuda_metrics": torch_cuda_metrics or {},
        },
        "manual_baseline": {
            "method": manual["method"],
            "pass_rate": manual["pass_rate"],
        },
        "aerorl_stack": {
            "pass_rate": aerorl["pass_rate"],
            "average_reward": aerorl["average_reward"],
            "component_averages": aerorl["component_averages"],
            "weights": aerorl["weights"],
        },
        "improvement": {
            "manual_false_passes_caught_count": len(false_passes_caught),
            "manual_false_pass_rate_among_manual_passes": round(
                len(false_passes_caught) / max(len(manual_pass_ids), 1), 6
            ),
            "quality_dimensions_checked": {
                "manual": 1,
                "aerorl": 4,
                "multiplier": 4.0,
            },
            "sample_false_pass_ids": false_passes_caught[:10],
            "best_examples": aerorl["best_examples"],
            "worst_examples": aerorl["worst_examples"],
        },
    }


def run_model_generated_benchmark(
    *,
    model_id: str,
    cache_dir: str | None,
    device: str,
    limit_docvqa: int,
    limit_chartqa: int,
    max_new_tokens: int,
    batch_size: int = 1,
    gpu_sample_interval_sec: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    gpu_before = _gpu_status()
    physical_gpu_index = _resolve_physical_gpu_index(device=device)
    visible_gpu_index = _resolve_visible_gpu_index(device)
    cuda_device = torch.device(device) if device.startswith("cuda") and torch.cuda.is_available() else None

    sampler = GPUSampler(interval_sec=gpu_sample_interval_sec) if cuda_device is not None else None
    if cuda_device is not None:
        torch.cuda.set_device(visible_gpu_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(visible_gpu_index)
        sampler.start()

    try:
        load_started = time.perf_counter()
        samples = load_samples(limit_docvqa=limit_docvqa, limit_chartqa=limit_chartqa)
        processor, model = _load_model(model_id=model_id, cache_dir=cache_dir, device=device)
        if cuda_device is not None:
            torch.cuda.synchronize(visible_gpu_index)
        load_elapsed = time.perf_counter() - load_started

        records: list[dict[str, Any]] = []
        total_generated_tokens = 0
        batch_latency_values: list[float] = []
        sample_latency_values: list[float] = []

        generate_started = time.perf_counter()
        for batch in _iter_sample_batches(samples, batch_size=batch_size):
            prompts = [_build_chat_prompt(sample.prompt) for sample in batch]
            texts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

            start = time.perf_counter()
            inputs = processor(text=texts, images=[sample.image for sample in batch], padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            if cuda_device is not None:
                torch.cuda.synchronize(visible_gpu_index)
            batch_latency_ms = (time.perf_counter() - start) * 1000.0
            sample_latency_ms = batch_latency_ms / max(len(batch), 1)

            decoded_texts, generated_token_counts = _decode_generated_texts(
                processor,
                output_ids,
                inputs.get("input_ids"),
            )

            batch_latency_values.append(batch_latency_ms)
            sample_latency_values.extend([sample_latency_ms] * len(batch))
            total_generated_tokens += sum(generated_token_counts)

            for sample, decoded, generated_tokens in zip(batch, decoded_texts, generated_token_counts):
                normalized_response, parsed_answer, was_json = _response_to_json_answer(decoded)
                reference_value: str | list[str]
                if len(sample.references) == 1:
                    reference_value = sample.references[0]
                else:
                    reference_value = list(sample.references)

                records.append(
                    {
                        "id": sample.sample_id,
                        "prompt": sample.prompt,
                        "response": normalized_response,
                        "reference": reference_value,
                        "metadata": {
                            "evidence_entities": list(sample.references),
                            "claimed_entities": [parsed_answer] if parsed_answer else [],
                            "latency_ms": round(sample_latency_ms, 3),
                            "batch_latency_ms": round(batch_latency_ms, 3),
                            "batch_size": len(batch),
                            "generated_tokens": int(generated_tokens),
                            "dataset": sample.dataset,
                            "raw_response": decoded,
                            "raw_json": was_json,
                            "empty_generation": decoded == "",
                        },
                    }
                )

        if cuda_device is not None:
            torch.cuda.synchronize(visible_gpu_index)
        generate_elapsed = time.perf_counter() - generate_started
    finally:
        gpu_samples = sampler.stop() if sampler is not None else []

    gpu_after = _gpu_status()
    gpu_profile = _summarize_gpu_samples(samples=gpu_samples, physical_gpu_index=physical_gpu_index)

    torch_cuda_metrics: dict[str, Any] = {}
    if cuda_device is not None:
        torch_cuda_metrics = {
            "peak_allocated_gb": round(float(torch.cuda.max_memory_allocated(visible_gpu_index)) / (1024**3), 3),
            "peak_reserved_gb": round(float(torch.cuda.max_memory_reserved(visible_gpu_index)) / (1024**3), 3),
        }

    stack = build_reward_stack(
        weights={"verifier": 0.45, "grounding": 0.3, "format": 0.2, "cost": 0.05},
        require_json=True,
        regex_pattern=r"^\{.*\}$",
        target_tokens=32,
        latency_budget_ms=1200.0,
    )
    aerorl = evaluate_records(records, reward_stack=stack, pass_threshold=0.5, top_k=5)
    manual = manual_baseline(records)

    report = build_model_generated_report(
        records=records,
        aerorl=aerorl,
        manual=manual,
        model_id=model_id,
        device=device,
        generate_elapsed_sec=generate_elapsed,
        load_elapsed_sec=load_elapsed,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
        prompt_style="single-image QA with strict JSON answer key",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        total_generated_tokens=total_generated_tokens,
        batch_latency_ms=batch_latency_values,
        sample_latency_ms=sample_latency_values,
        gpu_profile=gpu_profile,
        torch_cuda_metrics=torch_cuda_metrics,
    )
    return report, records


def run_model_generated_matrix(
    *,
    model_specs: list[dict[str, Any]],
    cache_dir: str | None,
    device: str,
    limit_docvqa: int,
    limit_chartqa: int,
    max_new_tokens: int,
    gpu_sample_interval_sec: float,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for spec in model_specs:
        report, _ = run_model_generated_benchmark(
            model_id=str(spec["model_id"]),
            cache_dir=cache_dir,
            device=device,
            limit_docvqa=limit_docvqa,
            limit_chartqa=limit_chartqa,
            max_new_tokens=max_new_tokens,
            batch_size=int(spec["batch_size"]),
            gpu_sample_interval_sec=gpu_sample_interval_sec,
        )
        generation = report["generation"]
        gpu_profile = generation.get("gpu_profile", {})
        torch_cuda_metrics = generation.get("torch_cuda_metrics", {})
        runs.append(
            {
                "model_id": report["generation"]["model_id"],
                "batch_size": generation["batch_size"],
                "samples_per_sec": generation["samples_per_sec"],
                "generated_tokens_per_sec": generation["generated_tokens_per_sec"],
                "manual_pass_rate": report["manual_baseline"]["pass_rate"],
                "aerorl_pass_rate": report["aerorl_stack"]["pass_rate"],
                "average_reward": report["aerorl_stack"]["average_reward"],
                "manual_false_passes_caught_count": report["improvement"]["manual_false_passes_caught_count"],
                "avg_sample_latency_ms": generation["sample_latency_ms"]["avg"],
                "p95_sample_latency_ms": generation["sample_latency_ms"]["p95"],
                "avg_gpu_utilization_pct": gpu_profile.get("avg_utilization_gpu_pct", 0.0),
                "max_gpu_utilization_pct": gpu_profile.get("max_utilization_gpu_pct", 0.0),
                "max_gpu_memory_used_mib": gpu_profile.get("max_memory_used_mib", 0.0),
                "peak_allocated_gb": torch_cuda_metrics.get("peak_allocated_gb", 0.0),
                "peak_reserved_gb": torch_cuda_metrics.get("peak_reserved_gb", 0.0),
            }
        )

    best_quality = max(runs, key=lambda item: (item["aerorl_pass_rate"], item["average_reward"]))
    best_throughput = max(runs, key=lambda item: item["samples_per_sec"])
    best_gpu_utilization = max(runs, key=lambda item: item["avg_gpu_utilization_pct"])

    return {
        "dataset": {
            "name": "DocVQA + ChartQA (real images, model-generated outputs)",
            "docvqa_rows": limit_docvqa,
            "chartqa_rows": limit_chartqa,
            "total_records": limit_docvqa + limit_chartqa,
        },
        "generation": {
            "device": device,
            "max_new_tokens": max_new_tokens,
            "gpu_sample_interval_sec": gpu_sample_interval_sec,
        },
        "runs": runs,
        "leaders": {
            "best_quality": best_quality,
            "best_throughput": best_throughput,
            "best_gpu_utilization": best_gpu_utilization,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-generated reward benchmark on cached real datasets")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--models", default="", help="Comma-separated model specs. Optional '@batch' suffix per model.")
    parser.add_argument("--cache-dir", default="/pub7/neel2/.cache_hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-docvqa", type=int, default=200)
    parser.add_argument("--limit-chartqa", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--output", default="reports/reward-model-generated-benchmark-2026-03-24.json")
    parser.add_argument("--replay-output", default="reports/reward-model-generated-replay-2026-03-24.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.models.strip():
        report = run_model_generated_matrix(
            model_specs=_parse_model_specs(args.models, default_batch_size=args.batch_size),
            cache_dir=args.cache_dir,
            device=args.device,
            limit_docvqa=args.limit_docvqa,
            limit_chartqa=args.limit_chartqa,
            max_new_tokens=args.max_new_tokens,
            gpu_sample_interval_sec=args.gpu_sample_interval_sec,
        )
        rendered = json.dumps(report, indent=2)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    report, records = run_model_generated_benchmark(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        limit_docvqa=args.limit_docvqa,
        limit_chartqa=args.limit_chartqa,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        gpu_sample_interval_sec=args.gpu_sample_interval_sec,
    )

    rendered = json.dumps(report, indent=2)
    output_path.write_text(rendered + "\n", encoding="utf-8")

    replay_path = Path(args.replay_output)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(replay_path, records)

    print(rendered)


if __name__ == "__main__":
    main()
