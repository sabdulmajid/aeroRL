from benchmarks.vlm_grpo_benchmark import run_benchmark_matrix


def test_benchmark_matrix_runs() -> None:
    result = run_benchmark_matrix(
        model_names=["Qwen/Qwen2.5-VL-7B-Instruct", "llava-hf/llava-1.5-7b-hf"],
        steps=2,
        mode="synthetic",
        matrix_size=64,
    )

    assert len(result["runs"]) == 2
    assert result["avg_iters_per_sec"] > 0.0
