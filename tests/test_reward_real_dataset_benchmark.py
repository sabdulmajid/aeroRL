from benchmarks.reward_real_dataset_benchmark import build_real_dataset_report


def test_real_dataset_report_contains_key_metrics() -> None:
    records = [
        {
            "id": "ok",
            "prompt": "p",
            "response": '{"valid":true,"reason":"none"}',
            "reference": '{"valid":true,"reason":"none"}',
            "metadata": {
                "evidence_entities": ["valid:true", "reason:none"],
                "claimed_entities": ["valid:true", "reason:none"],
                "latency_ms": 100,
            },
        },
        {
            "id": "bad",
            "prompt": "p",
            "response": '{"valid":true,"reason":"none"}',
            "reference": '{"valid":true,"reason":"min_length"}',
            "metadata": {
                "evidence_entities": ["valid:true", "reason:min_length"],
                "claimed_entities": ["valid:true", "reason:none"],
                "latency_ms": 120,
            },
        },
    ]

    report = build_real_dataset_report(records)

    assert report["dataset_size"] == 2
    assert "manual_baseline" in report
    assert "aerorl_stack" in report
    assert "improvement" in report
    assert "false_passes_caught" in report["improvement"]
