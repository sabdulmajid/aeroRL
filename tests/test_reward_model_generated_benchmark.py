import torch
from PIL import Image

import benchmarks.reward_model_generated_benchmark as benchmark
from benchmarks.reward_model_generated_benchmark import (
    _decode_generated_text,
    _iter_sample_batches,
    _parse_model_specs,
    build_model_generated_report,
    manual_baseline,
)


def test_manual_baseline_counts_contains_reference() -> None:
    records = [
        {"id": "a", "response": '{"answer":"cat"}', "reference": "cat"},
        {"id": "b", "response": '{"answer":"dog"}', "reference": "bird"},
    ]

    summary = manual_baseline(records)

    assert summary["method"] == "manual_contains_reference"
    assert summary["pass_rate"] == 0.5
    assert summary["pass_ids"] == ["a"]


def test_manual_baseline_accepts_any_reference_alias() -> None:
    records = [
        {
            "id": "docvqa::1",
            "response": '{"answer":"april 15, 2014"}',
            "reference": ["april 15,2014", "april 15, 2014"],
        }
    ]

    summary = manual_baseline(records)

    assert summary["pass_rate"] == 1.0
    assert summary["pass_ids"] == ["docvqa::1"]


def test_load_samples_preserves_all_docvqa_answers(monkeypatch) -> None:
    class DummyTable:
        def to_pylist(self) -> list[dict[str, object]]:
            return [
                {
                    "id": "row-1",
                    "query": "When?",
                    "answers": ["april 15,2014", "april 15, 2014"],
                    "image": {"bytes": b"ignored"},
                }
            ]

    monkeypatch.setattr(benchmark, "_read_arrow_table", lambda _path: DummyTable())
    monkeypatch.setattr(benchmark, "_decode_image", lambda _payload: Image.new("RGB", (1, 1)))

    samples = benchmark.load_samples(limit_docvqa=1, limit_chartqa=0)

    assert len(samples) == 1
    assert samples[0].references == ("april 15,2014", "april 15, 2014")


def test_decode_generated_text_returns_empty_when_no_new_tokens() -> None:
    class DummyProcessor:
        def batch_decode(self, ids, skip_special_tokens: bool = True) -> list[str]:
            return [""] if ids.shape[-1] == 0 else ["decoded"]

    output_ids = torch.tensor([[1, 2, 3]])
    input_ids = torch.tensor([[1, 2, 3]])

    decoded = _decode_generated_text(DummyProcessor(), output_ids, input_ids)

    assert decoded == ""


def test_iter_sample_batches_splits_without_dropping_records() -> None:
    samples = [
        benchmark.Sample(
            sample_id=f"id-{idx}",
            dataset="docvqa",
            prompt=f"q-{idx}",
            references=("a",),
            image=Image.new("RGB", (1, 1)),
        )
        for idx in range(5)
    ]

    batches = list(_iter_sample_batches(samples, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [sample.sample_id for batch in batches for sample in batch] == [f"id-{idx}" for idx in range(5)]


def test_parse_model_specs_supports_per_model_batch_sizes() -> None:
    specs = _parse_model_specs(
        "HuggingFaceTB/SmolVLM-500M-Instruct@16,Qwen/Qwen2.5-VL-7B-Instruct@4,llava-hf/llava-1.5-7b-hf",
        default_batch_size=8,
    )

    assert specs == [
        {"model_id": "HuggingFaceTB/SmolVLM-500M-Instruct", "batch_size": 16},
        {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "batch_size": 4},
        {"model_id": "llava-hf/llava-1.5-7b-hf", "batch_size": 8},
    ]


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
        batch_size=8,
        total_generated_tokens=64,
        batch_latency_ms=[100.0, 120.0],
        sample_latency_ms=[12.5, 15.0],
        gpu_profile={"avg_utilization_gpu_pct": 80.0, "max_memory_used_mib": 2048.0},
        torch_cuda_metrics={"peak_allocated_gb": 4.2},
    )

    assert report["dataset"]["total_records"] == 2
    assert report["dataset"]["sources"]["docvqa_rows"] == 1
    assert report["dataset"]["sources"]["chartqa_rows"] == 1
    assert report["improvement"]["manual_false_passes_caught_count"] == 1
    assert "aerorl_stack" in report
    assert report["generation"]["batch_size"] == 8
    assert report["generation"]["generated_tokens_total"] == 64
    assert report["generation"]["gpu_profile"]["avg_utilization_gpu_pct"] == 80.0
