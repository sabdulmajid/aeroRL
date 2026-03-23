from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict

from aerorl import AeroRLConfig, wrap_vlm_for_rl


def run_benchmark(model_name: str, steps: int = 20) -> dict:
    config = AeroRLConfig()
    model, ref_model = wrap_vlm_for_rl(model_name, config)

    start = time.perf_counter()
    random.seed(7)

    for _ in range(steps):
        time.sleep(0.01)

    elapsed = time.perf_counter() - start
    throughput = steps / elapsed

    return {
        "model": model_name,
        "steps": steps,
        "elapsed_sec": round(elapsed, 4),
        "iters_per_sec": round(throughput, 4),
        "config": asdict(config),
        "model_runtime": model["runtime"],
        "ref_runtime": ref_model["runtime"],
        "note": "Synthetic smoke benchmark; replace with real VLM GRPO benchmark measurements.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AeroRL synthetic GRPO benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    results = run_benchmark(args.model, steps=args.steps)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
