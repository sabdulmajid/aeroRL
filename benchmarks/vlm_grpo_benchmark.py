from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict

import torch

from aerorl import AeroRLConfig, wrap_vlm_for_rl


def _run_synthetic(steps: int) -> tuple[float, float]:
    start = time.perf_counter()
    random.seed(7)
    for _ in range(steps):
        time.sleep(0.01)
    elapsed = time.perf_counter() - start
    return elapsed, steps / elapsed


def _run_real_torch(steps: int, matrix_size: int) -> tuple[float, float, float, str]:
    if not torch.cuda.is_available():
        start = time.perf_counter()
        x = torch.randn(matrix_size, matrix_size)
        y = torch.randn(matrix_size, matrix_size)
        for _ in range(steps):
            _ = x @ y
        elapsed = time.perf_counter() - start
        return elapsed, steps / elapsed, 0.0, "cpu"

    device = torch.device("cuda")
    x = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    y = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(steps):
        _ = x @ y
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024**3)
    return elapsed, steps / elapsed, peak_vram_gb, str(device)


def run_benchmark(model_name: str, steps: int = 20, mode: str = "synthetic", matrix_size: int = 1024) -> dict:
    config = AeroRLConfig()
    model, ref_model = wrap_vlm_for_rl(model_name, config)

    if mode == "real":
        elapsed, throughput, peak_vram_gb, device = _run_real_torch(steps=steps, matrix_size=matrix_size)
        note = "Real torch benchmark path with measured throughput and peak memory."
    else:
        elapsed, throughput = _run_synthetic(steps=steps)
        peak_vram_gb = 0.0
        device = "synthetic"
        note = "Synthetic smoke benchmark path."

    return {
        "model": model_name,
        "steps": steps,
        "mode": mode,
        "device": device,
        "elapsed_sec": round(elapsed, 4),
        "iters_per_sec": round(throughput, 4),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "config": asdict(config),
        "model_runtime": model["runtime"],
        "ref_runtime": ref_model["runtime"],
        "trainer_backend": model["trainer"],
        "reference_quant": {
            "precision": ref_model["precision"],
            "quantization_mode": ref_model["quantization_mode"],
        },
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AeroRL benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--matrix-size", type=int, default=1024)
    args = parser.parse_args()

    results = run_benchmark(args.model, steps=args.steps, mode=args.mode, matrix_size=args.matrix_size)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
