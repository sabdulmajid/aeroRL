from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc

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


def _first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value)


def _pattern_from_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:6], 16) % 100
    if bucket < 58:
        return "good_json_exact"
    if bucket < 72:
        return "format_bad_contains"
    if bucket < 84:
        return "grounding_hidden"
    if bucket < 95:
        return "wrong_answer"
    return "verbose_slow"


def _build_record(sample_id: str, prompt: str, reference: str) -> dict[str, Any]:
    pattern = _pattern_from_id(sample_id)
    reference_clean = reference.strip()

    if pattern == "good_json_exact":
        response = json.dumps({"answer": reference_clean})
        claimed = [reference_clean]
        latency_ms = 110
    elif pattern == "format_bad_contains":
        response = f"answer: {reference_clean}"
        claimed = [reference_clean]
        latency_ms = 150
    elif pattern == "grounding_hidden":
        response = json.dumps({"answer": reference_clean})
        claimed = [f"not-{reference_clean}"]
        latency_ms = 170
    elif pattern == "wrong_answer":
        response = json.dumps({"answer": f"wrong-{reference_clean}"})
        claimed = [f"wrong-{reference_clean}"]
        latency_ms = 190
    else:
        response = json.dumps(
            {
                "answer": reference_clean,
                "reasoning": (
                    "This is an unnecessarily verbose response designed to exceed cost budgets "
                    "even when the final answer token is correct."
                ),
            }
        )
        claimed = [reference_clean]
        latency_ms = 1400

    return {
        "id": sample_id,
        "prompt": prompt,
        "response": response,
        "reference": reference_clean,
        "metadata": {
            "evidence_entities": [reference_clean],
            "claimed_entities": claimed,
            "latency_ms": latency_ms,
            "pattern": pattern,
        },
    }


def load_real_dataset_records(limit: int = 50000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    doc = _read_arrow_table(DOCVQA_TRAIN)
    doc_rows = doc.to_pylist()
    for row in doc_rows:
        if len(records) >= limit:
            return records
        sample_id = f"docvqa::{row['id']}"
        prompt = str(row.get("query", ""))
        reference = _first_text(row.get("answers", row.get("answer", "")))
        records.append(_build_record(sample_id=sample_id, prompt=prompt, reference=reference))

    for idx, path in enumerate(CHARTQA_TRAIN_FILES):
        table = _read_arrow_table(path)
        rows = table.to_pylist()
        for row_idx, row in enumerate(rows):
            if len(records) >= limit:
                return records
            sample_id = f"chartqa::{idx}:{row_idx}"
            prompt = str(row.get("query", ""))
            reference = str(row.get("label", ""))
            records.append(_build_record(sample_id=sample_id, prompt=prompt, reference=reference))

    return records


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


def run_large_scale_benchmark(limit: int = 50000) -> dict[str, Any]:
    gpu = _gpu_status()
    started = time.perf_counter()
    records = load_real_dataset_records(limit=limit)
    load_elapsed = time.perf_counter() - started

    stack = build_reward_stack(
        weights={"verifier": 0.45, "grounding": 0.3, "format": 0.2, "cost": 0.05},
        require_json=True,
        regex_pattern=r"^\{.*\}$",
        target_tokens=32,
        latency_budget_ms=400,
    )

    eval_started = time.perf_counter()
    aerorl = evaluate_records(records, reward_stack=stack, pass_threshold=0.5, top_k=5)
    eval_elapsed = time.perf_counter() - eval_started

    manual = manual_baseline(records)

    manual_pass_ids = set(manual["pass_ids"])
    aerorl_fail_ids = {
        item["id"]
        for item in aerorl["results"]
        if item["reward"]["total_reward"] < aerorl["pass_threshold"]
    }
    false_passes_caught = sorted(manual_pass_ids.intersection(aerorl_fail_ids))

    patterns: dict[str, int] = {}
    doc_count = 0
    chart_count = 0
    for row in records:
        pattern = str(row["metadata"].get("pattern", "unknown"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
        sample_id = str(row.get("id", ""))
        if sample_id.startswith("docvqa::"):
            doc_count += 1
        elif sample_id.startswith("chartqa::"):
            chart_count += 1

    compact = {
        "dataset": {
            "name": "DocVQA train + ChartQA train (cached local HF datasets)",
            "total_records": len(records),
            "sources": {
                "docvqa_train_rows": doc_count,
                "chartqa_train_rows": chart_count,
            },
            "synthetic_response_protocol": "deterministic corruption patterns over real prompts/references",
            "pattern_counts": patterns,
        },
        "compute": {
            "gpu_status_before_run": gpu,
            "execution_mode": "cpu",
            "data_load_sec": round(load_elapsed, 3),
            "evaluation_sec": round(eval_elapsed, 3),
            "records_per_sec": round((len(records) / eval_elapsed) if eval_elapsed else 0.0, 2),
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
            "worst_examples": aerorl["worst_examples"],
        },
    }
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run large-scale real dataset reward value benchmark")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--output", default="reports/reward-large-scale-benchmark-2026-03-23.json")
    args = parser.parse_args()

    report = run_large_scale_benchmark(limit=args.limit)
    rendered = json.dumps(report, indent=2)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
