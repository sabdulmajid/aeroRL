from __future__ import annotations

from dataclasses import asdict
from typing import Any, Tuple

from .config import AeroRLConfig


def wrap_vlm_for_rl(model_name_or_path: str, config: AeroRLConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
    config.validate()
    model = {
        "model_name_or_path": model_name_or_path,
        "runtime": "trainable",
        "aerorl": asdict(config),
    }
    ref_model = {
        "model_name_or_path": model_name_or_path,
        "runtime": "reference",
        "precision": f"int{config.quant_ref_bits}",
    }
    return model, ref_model
