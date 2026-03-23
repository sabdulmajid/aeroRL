from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class QuantizedReferenceRuntime:
    model_name_or_path: str
    quant_bits: int
    quantization_mode: str
    backend: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "runtime": "reference",
            "precision": f"int{self.quant_bits}",
            "quantization_mode": self.quantization_mode,
            "backend": self.backend,
        }


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_quant_backend(preferred_backend: str = "auto") -> str:
    if preferred_backend in {"torch", "bitsandbytes", "torchao"}:
        if preferred_backend == "torch":
            return "torch"
        if preferred_backend == "bitsandbytes" and _has_module("bitsandbytes"):
            return "bitsandbytes"
        if preferred_backend == "torchao" and _has_module("torchao"):
            return "torchao"
        return "torch"

    if _has_module("bitsandbytes"):
        return "bitsandbytes"
    if _has_module("torchao"):
        return "torchao"
    return "torch"


def create_quantized_reference_runtime(
    model_name_or_path: str,
    quant_bits: int = 8,
    preferred_backend: str = "auto",
) -> QuantizedReferenceRuntime:
    quant_backend = resolve_quant_backend(preferred_backend)

    if quant_bits == 16:
        return QuantizedReferenceRuntime(
            model_name_or_path=model_name_or_path,
            quant_bits=16,
            quantization_mode="fp16-reference",
            backend=quant_backend,
        )

    if quant_bits == 8:
        return QuantizedReferenceRuntime(
            model_name_or_path=model_name_or_path,
            quant_bits=8,
            quantization_mode=f"int8-{quant_backend}",
            backend=quant_backend,
        )

    return QuantizedReferenceRuntime(
        model_name_or_path=model_name_or_path,
        quant_bits=4,
        quantization_mode=f"int4-{quant_backend}",
        backend=quant_backend,
    )
