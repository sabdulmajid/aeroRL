from __future__ import annotations

from dataclasses import asdict
from typing import Any, Tuple

from .adapters import resolve_trainer_backend
from .config import AeroRLConfig
from .quant_ref import create_quantized_reference_runtime


def wrap_vlm_for_rl(model_name_or_path: str, config: AeroRLConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
    config.validate()
    trainer_adapter = resolve_trainer_backend(config.trainer_backend)
    reference_runtime = create_quantized_reference_runtime(
        model_name_or_path=model_name_or_path,
        quant_bits=config.quant_ref_bits,
        preferred_backend="auto",
    )

    model = {
        "model_name_or_path": model_name_or_path,
        "runtime": "trainable",
        "aerorl": asdict(config),
        "trainer": trainer_adapter.as_dict(),
    }
    ref_model = reference_runtime.as_dict()
    return model, ref_model
