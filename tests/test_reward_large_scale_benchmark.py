from benchmarks.reward_large_scale_real_dataset_benchmark import run_large_scale_benchmark


def test_large_scale_benchmark_small_limit() -> None:
    report = run_large_scale_benchmark(limit=200)

    assert report["dataset"]["total_records"] == 200
    assert report["manual_baseline"]["pass_rate"] >= 0.0
    assert report["aerorl_stack"]["pass_rate"] >= 0.0
    assert "manual_false_passes_caught_count" in report["improvement"]
