import json
from pathlib import Path

from aerorl.rewards import build_reward_stack, evaluate_records


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


def test_reward_replay_evaluator_custom_stack_summary(tmp_path: Path) -> None:
    records = [
        {
            "id": "sample-2",
            "prompt": "Return JSON",
            "response": '{"answer":"cat"}',
            "reference": '{"answer":"cat"}',
            "metadata": {"evidence_entities": ["cat"], "claimed_entities": ["cat"], "latency_ms": 30},
        }
    ]

    input_path = tmp_path / "samples2.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    loaded = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    stack = build_reward_stack(require_json=True, regex_pattern=r"^\{.*\}$", weights={"format": 0.5})
    summary = evaluate_records(loaded, reward_stack=stack, pass_threshold=0.8, top_k=1)

    assert summary["count"] == 1
    assert "component_averages" in summary
    assert summary["pass_rate"] == 1.0
