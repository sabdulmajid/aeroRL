import json
from pathlib import Path

from aerorl.rewards import evaluate_records


def test_reward_replay_evaluator_compatible_records(tmp_path: Path) -> None:
    records = [
        {
            "id": "sample-1",
            "prompt": "What word is visible?",
            "response": "STOP",
            "reference": "stop",
            "metadata": {"evidence_entities": ["stop"], "claimed_entities": ["stop"], "latency_ms": 25},
        }
    ]

    input_path = tmp_path / "samples.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    loaded = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    summary = evaluate_records(loaded)

    assert summary["count"] == 1
    assert summary["results"][0]["id"] == "sample-1"
