from benchmarks.reward_value_benchmark import build_value_report


def test_reward_value_benchmark_report_has_improvement_metrics() -> None:
    records = [
        {
            "id": "ok",
            "prompt": "p",
            "response": "{\"answer\":\"red\"}",
            "reference": "red",
            "metadata": {"evidence_entities": ["red"], "claimed_entities": ["red"], "latency_ms": 50},
        },
        {
            "id": "hidden-failure",
            "prompt": "p",
            "response": "{\"answer\":\"nike\"}",
            "reference": "nike",
            "metadata": {"evidence_entities": ["adidas"], "claimed_entities": ["nike"], "latency_ms": 70},
        },
    ]

    report = build_value_report(records)

    assert report["dataset_size"] == 2
    assert "manual_false_passes_caught" in report["improvement"]
    assert "quality_gate_gain" in report["improvement"]
