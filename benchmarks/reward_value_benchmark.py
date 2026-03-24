from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerorl import build_reward_stack, evaluate_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manual_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    contains_pass = 0

    for row in records:
        response = str(row.get("response", "")).strip().lower()
        reference = str(row.get("reference", "")).strip().lower()
        passed = bool(reference) and reference in response
        contains_pass += int(passed)
        scored.append({"id": row.get("id"), "pass": passed})

    total = len(records)
    pass_rate = contains_pass / total if total else 0.0
    return {
        "count": total,
        "method": "manual_contains_reference",
        "pass_rate": round(pass_rate, 6),
        "results": scored,
    }


def aerorl_stack(records: list[dict[str, Any]]) -> dict[str, Any]:
    stack = build_reward_stack(
        weights={"verifier": 0.45, "grounding": 0.3, "format": 0.2, "cost": 0.05},
        require_json=True,
        regex_pattern=r"^\{.*\}$",
        target_tokens=32,
        latency_budget_ms=400,
    )
    return evaluate_records(records, reward_stack=stack, pass_threshold=0.5, top_k=3)


def summarize_diagnostics(aerorl_summary: dict[str, Any]) -> dict[str, int]:
    counts = {
        "format_issues": 0,
        "grounding_issues": 0,
        "verifier_issues": 0,
        "cost_issues": 0,
    }

    for row in aerorl_summary.get("results", []):
        components = row["reward"]["components"]
        if components.get("format", 0.0) < 0:
            counts["format_issues"] += 1
        if components.get("grounding", 0.0) < 0:
            counts["grounding_issues"] += 1
        if components.get("verifier", 0.0) < 0:
            counts["verifier_issues"] += 1
        if components.get("cost", 0.0) < 0.5:
            counts["cost_issues"] += 1

    return counts


def build_value_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    manual = manual_baseline(records)
    aerorl = aerorl_stack(records)
    diagnostics = summarize_diagnostics(aerorl)

    manual_pass_ids = {item["id"] for item in manual["results"] if item["pass"]}
    aerorl_fail_ids = {
        item["id"]
        for item in aerorl["results"]
        if item["reward"]["total_reward"] < aerorl.get("pass_threshold", 0.5)
    }
    hidden_failures_caught = sorted(manual_pass_ids.intersection(aerorl_fail_ids))

    manual_only_issues = {
        "false_pass_rate": round(len(hidden_failures_caught) / max(len(manual_pass_ids), 1), 6),
        "false_pass_count": len(hidden_failures_caught),
    }

    observability_gain = {
        "manual_dimensions": 1,
        "aerorl_dimensions": 4,
        "multiplier": 4.0,
    }

    return {
        "dataset_size": len(records),
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
        "diagnostics_found": diagnostics,
        "improvement": {
            "pass_rate_delta": round(aerorl["pass_rate"] - manual["pass_rate"], 6),
            "observability_gain": observability_gain,
            "manual_false_passes_caught": hidden_failures_caught,
            "quality_gate_gain": manual_only_issues,
            "top_failure_examples": aerorl["worst_examples"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manual reward scoring vs AeroRL reward stack")
    parser.add_argument("--input", default="examples/reward_value_benchmark_dataset.jsonl")
    parser.add_argument("--output", default="reports/reward-value-benchmark-2026-03-23.json")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    report = build_value_report(records)
    rendered = json.dumps(report, indent=2)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
