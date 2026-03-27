from pathlib import Path

from benchmarks.lmms_eval_standard_benchmark import (
    _build_command,
    _primary_metric_entry,
    _summarize_lmms_results,
)


def test_primary_metric_entry_prefers_task_primary_metric() -> None:
    metric = _primary_metric_entry(
        {
            "samples": 500,
            "relaxed_human_split,none": 0.6,
            "relaxed_overall,none": 0.72,
            "relaxed_overall_stderr,none": 0.01,
        }
    )

    assert metric == ("relaxed_overall,none", 0.72)


def test_summarize_lmms_results_collects_task_scores() -> None:
    summary = _summarize_lmms_results(
        {
            "results": {
                "aerorl_public_vqa_lite": {
                    "alias": "aerorl_public_vqa_lite",
                    " ": " ",
                },
                "aerorl_docvqa_val_lite": {
                    "alias": "aerorl_docvqa_val_lite",
                    "anls,none": 0.61,
                    "anls_stderr,none": 0.02,
                },
                "aerorl_chartqa_lite": {
                    "alias": "aerorl_chartqa_lite",
                    "relaxed_overall,none": 0.73,
                    "relaxed_human_split,none": 0.74,
                },
            },
            "n-samples": {
                "aerorl_docvqa_val_lite": {"original": 500, "effective": 500},
                "aerorl_chartqa_lite": {"original": 500, "effective": 500},
            },
        }
    )

    assert summary["total_samples"] == 1000
    assert summary["mean_primary_score"] == 0.67
    assert summary["tasks"][0]["primary_metric"] == "relaxed_overall,none"
    assert summary["tasks"][1]["primary_metric"] == "anls,none"


def test_build_command_includes_repo_task_wrappers() -> None:
    command = _build_command(
        model="qwen2_vl",
        pretrained="Qwen/Qwen2-VL-2B-Instruct",
        tasks="aerorl_public_vqa_lite",
        include_path=Path("/tmp/lmms_tasks"),
        device="cuda:0",
        batch_size=1,
        output_path=Path("/tmp/out"),
        verbosity="WARNING",
        limit=None,
        process_with_media=True,
        force_simple=False,
        extra_model_args="",
    )

    assert command[:4] == ["python", "-m", "lmms_eval", "eval"]
    assert "--include_path" in command
    assert "/tmp/lmms_tasks" in command
    assert "--process_with_media" in command
