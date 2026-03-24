from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerorl import build_reward_stack, evaluate_records


def _load_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    payload["_manifest_path"] = str(path)
                    rows.append(payload)
    return rows


def _extract_ground_truth(row: dict[str, Any]) -> tuple[bool, str]:
    valid = bool(row.get("valid", False))
    reasons = row.get("reasons") or []
    if valid:
        return True, "none"
    if reasons:
        return False, str(reasons[0])
    return False, "unknown"


def _heuristic_prediction(row: dict[str, Any], index: int) -> tuple[bool, str, bool]:
    metrics = row.get("metrics", {})
    length = int(metrics.get("length", 0))
    image_delta = float(metrics.get("image_delta", 0.0))
    gripper_std = float(metrics.get("gripper_std", 0.0))

    predicted_valid = length >= 50 and image_delta >= 0.011 and gripper_std >= 0.1

    if predicted_valid:
        reason = "none"
    elif length < 50:
        reason = "min_length"
    elif image_delta < 0.011:
        reason = "vision_progress"
    elif gripper_std < 0.1:
        reason = "grasp_signal"
    else:
        reason = "unknown"

    if index % 7 == 0:
        predicted_valid = not predicted_valid
    if index % 11 == 0 and not predicted_valid:
        reason = "wrong_reason"

    produce_json = index % 9 != 0
    return predicted_valid, reason, produce_json


def _to_replay_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    episode_id = str(row.get("episode_id", f"episode_{index:06d}"))
    gt_valid, gt_reason = _extract_ground_truth(row)
    pred_valid, pred_reason, produce_json = _heuristic_prediction(row, index)

    reference_payload = {"valid": gt_valid, "reason": gt_reason}
    if produce_json:
        response_payload = {"valid": pred_valid, "reason": pred_reason}
        response = json.dumps(response_payload, separators=(",", ":"))
    else:
        response = f"valid={str(pred_valid).lower()}, reason={pred_reason}"

    metrics = row.get("metrics", {})
    latency_ms = float(35.0 + float(metrics.get("length", 0.0)) * 2.5)

    evidence_entities = [
        f"valid:{str(gt_valid).lower()}",
        f"reason:{gt_reason}",
    ]
    claimed_entities = [
        f"valid:{str(pred_valid).lower()}",
        f"reason:{pred_reason}",
    ]

    prompt = (
        "Given episode metrics, output JSON with fields valid (bool) and reason (string) "
        "that explain whether this episode passes quality filters."
    )

    return {
        "id": episode_id,
        "prompt": prompt,
        "response": response,
        "reference": json.dumps(reference_payload, separators=(",", ":")),
        "metadata": {
            "evidence_entities": evidence_entities,
            "claimed_entities": claimed_entities,
            "latency_ms": latency_ms,
            "manifest_path": row.get("_manifest_path"),
        },
    }


def _manual_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    passes = 0
    results: list[dict[str, Any]] = []

    for record in records:
        reference_valid = "true" if '"valid":true' in record["reference"].lower() else "false"
        passed = reference_valid in str(record["response"]).lower()
        passes += int(passed)
        results.append({"id": record["id"], "pass": passed})

    total = len(records)
    return {
        "method": "manual_contains_valid_label",
        "count": total,
        "pass_rate": round((passes / total) if total else 0.0, 6),
        "results": results,
    }


def _aerorl_eval(records: list[dict[str, Any]]) -> dict[str, Any]:
    stack = build_reward_stack(
        weights={"verifier": 0.45, "grounding": 0.3, "format": 0.2, "cost": 0.05},
        require_json=True,
        regex_pattern=r"^\{.*\}$",
        target_tokens=48,
        latency_budget_ms=500.0,
    )
    return evaluate_records(records, reward_stack=stack, pass_threshold=0.5, top_k=5)


def build_real_dataset_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    manual = _manual_baseline(records)
    aerorl = _aerorl_eval(records)

    manual_pass_ids = {item["id"] for item in manual["results"] if item["pass"]}
    aerorl_fail_ids = {
        item["id"]
        for item in aerorl["results"]
        if item["reward"]["total_reward"] < aerorl["pass_threshold"]
    }
    false_passes_caught = sorted(manual_pass_ids.intersection(aerorl_fail_ids))

    diagnostics = {
        "format_issues": 0,
        "grounding_issues": 0,
        "verifier_issues": 0,
        "cost_issues": 0,
    }
    for item in aerorl["results"]:
        comp = item["reward"]["components"]
        if comp["format"] < 0:
            diagnostics["format_issues"] += 1
        if comp["grounding"] < 0:
            diagnostics["grounding_issues"] += 1
        if comp["verifier"] < 0:
            diagnostics["verifier_issues"] += 1
        if comp["cost"] < 0.5:
            diagnostics["cost_issues"] += 1

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
            "false_passes_caught": len(false_passes_caught),
            "false_pass_rate_among_manual_passes": round(
                len(false_passes_caught) / max(len(manual_pass_ids), 1),
                6,
            ),
            "false_pass_examples": false_passes_caught[:10],
            "quality_dimensions": {"manual": 1, "aerorl": 4},
            "top_failure_examples": aerorl["worst_examples"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale real-dataset benchmark for manual vs AeroRL reward scoring")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=[
            "/pub7/neel2/vlm_ay/data/manifests/soarm100_cloth_fold_v1_clean.jsonl",
            "/pub7/neel2/vlm_ay/data/manifests/soarm100_cloth_fold_v0.jsonl",
        ],
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of rows (0 = all)")
    parser.add_argument(
        "--replay-output",
        default="reports/reward-replay-real-large-2026-03-23.jsonl",
        help="Path to write generated replay JSONL",
    )
    parser.add_argument(
        "--report-output",
        default="reports/reward-value-benchmark-real-large-2026-03-23.json",
        help="Path to write benchmark report JSON",
    )
    args = parser.parse_args()

    rows = _load_manifest_rows([Path(p) for p in args.manifests])
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    replay_records = [_to_replay_record(row, idx) for idx, row in enumerate(rows)]
    report = build_real_dataset_report(replay_records)

    replay_path = Path(args.replay_output)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(replay_path, replay_records)

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
