from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_SUITE = "public_vqa_lite"
DEFAULT_SUITE_TASK = "aerorl_public_vqa_lite"
PRIMARY_METRIC_PRIORITY = (
    "anls,none",
    "relaxed_overall,none",
    "acc,none",
    "exact_match,none",
)


class GPUSampler:
    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = max(interval_sec, 0.1)
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self._capture_once()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec * 4)
            self._thread = None
        self._capture_once()
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._capture_once()

    def _capture_once(self) -> None:
        status = _gpu_status()
        if status.get("available"):
            self.samples.append(
                {
                    "timestamp_sec": round(time.time(), 3),
                    "gpus": [dict(gpu) for gpu in status.get("gpus", [])],
                }
            )


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _gpu_status() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    gpus: list[dict[str, Any]] = []
    for line in out:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "utilization_gpu_pct": int(fields[4]),
                "power_draw_w": _safe_float(fields[5]),
            }
        )
    return {"available": True, "gpus": gpus}


def _resolve_visible_gpu_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    return int(device.split(":", 1)[1])


def _resolve_physical_gpu_index(device: str) -> int | None:
    visible_index = _resolve_visible_gpu_index(device)
    if visible_index is None:
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return visible_index

    visible_ids = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if visible_index >= len(visible_ids):
        return None
    if visible_ids[visible_index].isdigit():
        return int(visible_ids[visible_index])
    return None


def _summarize_gpu_samples(samples: list[dict[str, Any]], physical_gpu_index: int | None) -> dict[str, Any]:
    if physical_gpu_index is None:
        return {}

    selected: list[dict[str, Any]] = []
    for sample in samples:
        gpu = next((item for item in sample.get("gpus", []) if item.get("index") == physical_gpu_index), None)
        if gpu is not None:
            selected.append(gpu)

    if not selected:
        return {"physical_gpu_index": physical_gpu_index, "sample_count": 0}

    memory_values = [float(item["memory_used_mib"]) for item in selected]
    utilization_values = [float(item["utilization_gpu_pct"]) for item in selected]
    power_values = [float(item["power_draw_w"]) for item in selected if item.get("power_draw_w") is not None]

    profile = {
        "physical_gpu_index": physical_gpu_index,
        "sample_count": len(selected),
        "avg_memory_used_mib": round(sum(memory_values) / len(memory_values), 3),
        "max_memory_used_mib": round(max(memory_values), 3),
        "avg_utilization_gpu_pct": round(sum(utilization_values) / len(utilization_values), 3),
        "max_utilization_gpu_pct": round(max(utilization_values), 3),
    }
    if power_values:
        profile["avg_power_draw_w"] = round(sum(power_values) / len(power_values), 3)
        profile["max_power_draw_w"] = round(max(power_values), 3)
    return profile


def _result_files(raw_output_dir: Path) -> tuple[Path, list[Path]]:
    result_files = sorted(raw_output_dir.rglob("*_results.json"))
    if not result_files:
        raise FileNotFoundError(f"No lmms-eval results file found in {raw_output_dir}")
    result_json = result_files[-1]
    run_prefix = result_json.name.replace("_results.json", "")
    sample_files = sorted(result_json.parent.glob(f"{run_prefix}_samples_*.jsonl"))
    return result_json, sample_files


def _primary_metric_entry(task_result: dict[str, Any]) -> tuple[str, float] | None:
    numeric_metrics = {
        key: float(value)
        for key, value in task_result.items()
        if isinstance(value, (int, float))
        and key not in {"samples"}
        and "stderr" not in key
        and not key.endswith(",none_stderr")
    }
    for key in PRIMARY_METRIC_PRIORITY:
        if key in numeric_metrics:
            return key, numeric_metrics[key]
    if not numeric_metrics:
        return None
    first_key = sorted(numeric_metrics)[0]
    return first_key, numeric_metrics[first_key]


def _summarize_lmms_results(results_payload: dict[str, Any]) -> dict[str, Any]:
    task_results = results_payload.get("results", {})
    sample_counts = results_payload.get("n-samples", {})
    tasks: list[dict[str, Any]] = []
    primary_scores: list[float] = []
    total_samples = 0

    for task_name, task_result in sorted(task_results.items()):
        if not isinstance(task_result, dict):
            continue
        if " " in task_result and len(task_result) <= 2:
            continue
        sample_info = sample_counts.get(task_name, {})
        sample_count = int(sample_info.get("effective", task_result.get("samples", 0)))
        total_samples += sample_count
        metrics = {
            key: value
            for key, value in task_result.items()
            if key not in {"alias", "samples", " "} and "stderr" not in key
        }
        primary_metric = _primary_metric_entry(task_result)
        task_summary = {
            "task": task_name,
            "samples": sample_count,
            "metrics": metrics,
        }
        if primary_metric is not None:
            task_summary["primary_metric"] = primary_metric[0]
            task_summary["primary_score"] = round(primary_metric[1], 6)
            primary_scores.append(primary_metric[1])
        tasks.append(task_summary)

    return {
        "total_samples": total_samples,
        "mean_primary_score": round(sum(primary_scores) / len(primary_scores), 6) if primary_scores else 0.0,
        "tasks": tasks,
    }


def _build_command(
    *,
    model: str,
    pretrained: str,
    tasks: str,
    include_path: Path,
    device: str,
    batch_size: int,
    output_path: Path,
    verbosity: str,
    limit: int | None,
    process_with_media: bool,
    force_simple: bool,
    extra_model_args: str,
) -> list[str]:
    model_args = [f"pretrained={pretrained}"]
    if extra_model_args.strip():
        model_args.append(extra_model_args.strip())

    cmd = [
        "python",
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        model,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        tasks,
        "--include_path",
        str(include_path),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
        "--log_samples",
        "--verbosity",
        verbosity,
    ]
    if process_with_media:
        cmd.append("--process_with_media")
    if force_simple:
        cmd.append("--force_simple")
    if limit is not None and limit >= 0:
        cmd.extend(["--limit", str(limit)])
    return cmd


def run_standard_benchmark(
    *,
    model: str,
    pretrained: str,
    suite: str,
    tasks: str,
    include_path: Path,
    cache_dir: str,
    device: str,
    batch_size: int,
    output_path: Path,
    raw_output_dir: Path,
    log_path: Path,
    verbosity: str,
    limit: int | None,
    gpu_sample_interval_sec: float,
    process_with_media: bool,
    force_simple: bool,
    extra_model_args: str,
) -> dict[str, Any]:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = _build_command(
        model=model,
        pretrained=pretrained,
        tasks=tasks,
        include_path=include_path,
        device=device,
        batch_size=batch_size,
        output_path=raw_output_dir,
        verbosity=verbosity,
        limit=limit,
        process_with_media=process_with_media,
        force_simple=force_simple,
        extra_model_args=extra_model_args,
    )

    env = os.environ.copy()
    env["HF_HOME"] = cache_dir
    env["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
    env["TRANSFORMERS_CACHE"] = str(Path(cache_dir) / "hub")

    gpu_before = _gpu_status()
    physical_gpu_index = _resolve_physical_gpu_index(device=device)
    sampler = GPUSampler(interval_sec=gpu_sample_interval_sec) if device.startswith("cuda") else None
    if sampler is not None:
        sampler.start()

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    gpu_samples = sampler.stop() if sampler is not None else []
    gpu_after = _gpu_status()

    combined_output = completed.stdout
    if completed.stderr:
        combined_output += ("\n" if combined_output else "") + completed.stderr
    log_path.write_text(combined_output, encoding="utf-8")

    if completed.returncode != 0:
        tail = "\n".join(combined_output.splitlines()[-40:])
        raise RuntimeError(f"lmms-eval failed with exit code {completed.returncode}\n{tail}")

    result_json, sample_files = _result_files(raw_output_dir)
    results_payload = json.loads(result_json.read_text(encoding="utf-8"))
    results_summary = _summarize_lmms_results(results_payload)

    report = {
        "suite": {
            "name": suite,
            "tasks": tasks.split(","),
        },
        "generation": {
            "model_backend": model,
            "pretrained": pretrained,
            "device": device,
            "batch_size": batch_size,
            "elapsed_sec": round(elapsed, 3),
            "samples_per_sec": round(
                (results_summary["total_samples"] / elapsed) if elapsed else 0.0,
                3,
            ),
            "gpu_status_before": gpu_before,
            "gpu_status_after": gpu_after,
            "gpu_profile": _summarize_gpu_samples(gpu_samples, physical_gpu_index=physical_gpu_index),
        },
        "results": results_summary,
        "artifacts": {
            "aggregated_results_json": str(result_json),
            "sample_logs": [str(path) for path in sample_files],
            "raw_output_dir": str(raw_output_dir),
            "runner_log": str(log_path),
        },
        "command": {
            "argv": command,
            "shell": " ".join(shlex.quote(part) for part in command),
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standard public lmms-eval benchmark and summarize the results.")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--tasks", default=DEFAULT_SUITE_TASK)
    parser.add_argument("--model", default="qwen2_vl")
    parser.add_argument("--pretrained", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--cache-dir", default="/pub7/neel2/.cache_hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--verbosity", default="WARNING")
    parser.add_argument("--gpu-sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--process-with-media", action="store_true")
    parser.add_argument("--force-simple", action="store_true")
    parser.add_argument("--extra-model-args", default="")
    parser.add_argument("--output", default="reports/lmms-eval-standard-benchmark-2026-03-27.json")
    parser.add_argument("--raw-output-dir", default="reports/lmms-eval-standard-raw-2026-03-27")
    parser.add_argument("--log-output", default="reports/lmms-eval-standard-2026-03-27.log")
    args = parser.parse_args()

    include_path = Path(__file__).with_name("lmms_tasks")
    report = run_standard_benchmark(
        model=args.model,
        pretrained=args.pretrained,
        suite=args.suite,
        tasks=args.tasks,
        include_path=include_path,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        output_path=Path(args.output),
        raw_output_dir=Path(args.raw_output_dir),
        log_path=Path(args.log_output),
        verbosity=args.verbosity,
        limit=None if args.limit < 0 else args.limit,
        gpu_sample_interval_sec=args.gpu_sample_interval_sec,
        process_with_media=args.process_with_media,
        force_simple=args.force_simple,
        extra_model_args=args.extra_model_args,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
