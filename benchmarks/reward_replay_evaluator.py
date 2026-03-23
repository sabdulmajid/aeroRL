from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerorl.rewards import evaluate_records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline replay data with AeroRL reward stack")
    parser.add_argument("--input", required=True, help="Path to JSONL file with prompt/response/reference/metadata")
    parser.add_argument("--output", default="", help="Optional path to write JSON summary")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = _read_jsonl(input_path)
    summary = evaluate_records(records)
    rendered = json.dumps(summary, indent=2)

    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
