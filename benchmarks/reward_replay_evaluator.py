from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerorl.rewards import build_reward_stack, evaluate_records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _parse_weights(weight_args: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for entry in weight_args:
        if "=" not in entry:
            raise ValueError(f"Invalid weight entry '{entry}'. Expected format name=value")
        key, raw = entry.split("=", 1)
        parsed[key.strip()] = float(raw)
    return parsed


def _print_brief(summary: dict[str, Any]) -> None:
    print("\n=== Reward Summary ===")
    print(f"count: {summary['count']}")
    print(f"average_reward: {summary['average_reward']}")
    print(f"pass_rate: {summary['pass_rate']} (threshold={summary['pass_threshold']})")
    print("component_averages:")
    for name, value in summary["component_averages"].items():
        print(f"  - {name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline replay data with AeroRL reward stack")
    parser.add_argument("--input", required=True, help="Path to JSONL file with prompt/response/reference/metadata")
    parser.add_argument("--output", default="", help="Optional path to write JSON summary")
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Override component weight using name=value (repeatable). Example: --weight verifier=0.5",
    )
    parser.add_argument("--require-json", action="store_true", help="Enable JSON format validation in format reward")
    parser.add_argument("--regex-pattern", default="", help="Optional regex constraint used by format reward")
    parser.add_argument("--target-tokens", type=int, default=128, help="Token budget used by cost reward")
    parser.add_argument("--latency-budget-ms", type=float, default=500.0, help="Latency budget used by cost reward")
    parser.add_argument("--pass-threshold", type=float, default=0.3, help="Threshold used to compute pass_rate")
    parser.add_argument("--top-k", type=int, default=3, help="How many best/worst examples to include")
    parser.add_argument("--quiet", action="store_true", help="Print JSON only, without brief summary")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = _read_jsonl(input_path)
    stack = build_reward_stack(
        weights=_parse_weights(args.weight),
        require_json=args.require_json,
        regex_pattern=(args.regex_pattern or None),
        target_tokens=args.target_tokens,
        latency_budget_ms=args.latency_budget_ms,
    )
    summary = evaluate_records(
        records,
        reward_stack=stack,
        pass_threshold=args.pass_threshold,
        top_k=args.top_k,
    )
    rendered = json.dumps(summary, indent=2)

    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        _print_brief(summary)
    print(rendered)


if __name__ == "__main__":
    main()
