from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    reference: str
    image: Image.Image


def _gpu_status() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    gpus: list[dict[str, Any]] = []
    for line in out:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "utilization_gpu_pct": int(fields[4]),
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


def load_samples(limit_docvqa: int, limit_chartqa: int) -> list[Sample]:
    samples: list[Sample] = []

    if limit_docvqa > 0:
        doc = _read_arrow_table(DOCVQA_TRAIN)
        for row in doc.to_pylist()[:limit_docvqa]:
            prompt = _safe_text(row.get("query", "")).strip()
            reference = _safe_text(row.get("answers", row.get("answer", ""))).strip()
            if not prompt or not reference:
                continue
            samples.append(
                Sample(
                    sample_id=f"docvqa::{row.get('id', len(samples))}",
                    dataset="docvqa",
                    prompt=prompt,
                    reference=reference,
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
                reference = _safe_text(row.get("label", row.get("answer", ""))).strip()
                if not prompt or not reference:
                    continue
                samples.append(
                    Sample(
                        sample_id=f"chartqa::{shard_idx}:{row_idx}",
                        dataset="chartqa",
                        prompt=prompt,
                        reference=reference,
                        image=_decode_image(row.get("image")),
                    )
                )
                remaining -= 1

    return samples


def _load_model(model_id: str, cache_dir: str | None, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    resolved_model_id = _resolve_model_path(model_id=model_id, cache_dir=cache_dir)
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(cache_dir) / "hub")

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(resolved_model_id, cache_dir=cache_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
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


def _decode_generated_text(processor: Any, output_ids: Any, input_ids: Any | None) -> str:
    generated_ids = output_ids
    if input_ids is not None:
        try:
            generated_ids = output_ids[:, input_ids.shape[1] :]
        except Exception:
            generated_ids = output_ids

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    response = decoded[0].strip() if decoded else ""
    if response:
        return response

    full = processor.batch_decode(output_ids, skip_special_tokens=True)
    return full[0].strip() if full else ""


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


def manual_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = 0
    pass_ids: list[str] = []
    for row in records:
        response = str(row["response"]).lower()
        reference = str(row["reference"]).lower()
        ok = reference in response
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
            "max_new_tokens": max_new_tokens,
            "load_sec": round(load_elapsed_sec, 3),
            "generate_sec": round(generate_elapsed_sec, 3),
            "samples_per_sec": round((len(records) / generate_elapsed_sec) if generate_elapsed_sec else 0.0, 3),
            "gpu_status_before": gpu_before,
            "gpu_status_after": gpu_after,
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gpu_before = _gpu_status()

    load_started = time.perf_counter()
    samples = load_samples(limit_docvqa=limit_docvqa, limit_chartqa=limit_chartqa)
    processor, model = _load_model(model_id=model_id, cache_dir=cache_dir, device=device)
    load_elapsed = time.perf_counter() - load_started

    records: list[dict[str, Any]] = []
    generate_started = time.perf_counter()

    for sample in samples:
        prompt = (
            "Answer the question from the image. "
            "Return strict JSON only with one key: answer. "
            f"Question: {sample.prompt}"
        )

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        start = time.perf_counter()
        inputs = processor(text=text, images=[sample.image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        decoded = _decode_generated_text(processor, output_ids, inputs.get("input_ids"))
        normalized_response, parsed_answer, was_json = _response_to_json_answer(decoded)

        records.append(
            {
                "id": sample.sample_id,
                "prompt": sample.prompt,
                "response": normalized_response,
                "reference": sample.reference,
                "metadata": {
                    "evidence_entities": [sample.reference],
                    "claimed_entities": [parsed_answer] if parsed_answer else [],
                    "latency_ms": round(latency_ms, 3),
                    "dataset": sample.dataset,
                    "raw_response": decoded,
                    "raw_json": was_json,
                },
            }
        )

    generate_elapsed = time.perf_counter() - generate_started
    gpu_after = _gpu_status()

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
    )
    return report, records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-generated reward benchmark on cached real datasets")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--cache-dir", default="/pub7/neel2/.cache_hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-docvqa", type=int, default=200)
    parser.add_argument("--limit-chartqa", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", default="reports/reward-model-generated-benchmark-2026-03-24.json")
    parser.add_argument("--replay-output", default="reports/reward-model-generated-replay-2026-03-24.jsonl")
    args = parser.parse_args()

    report, records = run_model_generated_benchmark(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        limit_docvqa=args.limit_docvqa,
        limit_chartqa=args.limit_chartqa,
        max_new_tokens=args.max_new_tokens,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    output_path.write_text(rendered + "\n", encoding="utf-8")

    replay_path = Path(args.replay_output)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(replay_path, records)

    print(rendered)


if __name__ == "__main__":
    main()
