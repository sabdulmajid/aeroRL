from benchmarks.reward_model_generated_benchmark import build_model_generated_report, manual_baseline


def test_manual_baseline_counts_contains_reference() -> None:
    records = [
        {"id": "a", "response": '{"answer":"cat"}', "reference": "cat"},
        {"id": "b", "response": '{"answer":"dog"}', "reference": "bird"},
    ]

    summary = manual_baseline(records)

    assert summary["method"] == "manual_contains_reference"
    assert summary["pass_rate"] == 0.5
    assert summary["pass_ids"] == ["a"]


def test_model_generated_report_has_core_sections() -> None:
    records = [
        {"id": "docvqa::1", "response": '{"answer":"cat"}', "reference": "cat"},
        {"id": "chartqa::0:1", "response": '{"answer":"dog"}', "reference": "dog"},
    ]
    manual = {"method": "manual_contains_reference", "pass_rate": 1.0, "pass_ids": ["docvqa::1", "chartqa::0:1"]}
    aerorl = {
        "pass_rate": 0.5,
        "average_reward": 0.2,
        "component_averages": {"verifier": 0.0, "grounding": 0.0, "format": 1.0, "cost": 1.0},
        "weights": {"verifier": 0.45, "grounding": 0.3, "format": 0.2, "cost": 0.05},
        "pass_threshold": 0.5,
        "best_examples": [{"id": "docvqa::1", "total_reward": 1.0, "components": {}}],
        "worst_examples": [{"id": "chartqa::0:1", "total_reward": -0.1, "components": {}}],
        "results": [
            {"id": "docvqa::1", "reward": {"total_reward": 1.0, "components": {}}},
            {"id": "chartqa::0:1", "reward": {"total_reward": -0.1, "components": {}}},
        ],
    }

    report = build_model_generated_report(
        records=records,
        aerorl=aerorl,
        manual=manual,
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        device="cuda:0",
        generate_elapsed_sec=2.0,
        load_elapsed_sec=1.0,
        gpu_before={"available": True, "gpus": []},
        gpu_after={"available": True, "gpus": []},
        prompt_style="single-image QA with strict JSON answer key",
        max_new_tokens=24,
    )

    assert report["dataset"]["total_records"] == 2
    assert report["dataset"]["sources"]["docvqa_rows"] == 1
    assert report["dataset"]["sources"]["chartqa_rows"] == 1
    assert report["improvement"]["manual_false_passes_caught_count"] == 1
    assert "aerorl_stack" in report
