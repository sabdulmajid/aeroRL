from __future__ import annotations

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


def create_quantized_reference_runtime(model_name_or_path: str, quant_bits: int = 8) -> QuantizedReferenceRuntime:
    if quant_bits == 16:
        return QuantizedReferenceRuntime(
            model_name_or_path=model_name_or_path,
            quant_bits=16,
            quantization_mode="fp16-reference",
            backend="torch",
        )

    if quant_bits == 8:
        return QuantizedReferenceRuntime(
            model_name_or_path=model_name_or_path,
            quant_bits=8,
            quantization_mode="int8-ready",
            backend="torch",
        )

    return QuantizedReferenceRuntime(
        model_name_or_path=model_name_or_path,
        quant_bits=4,
        quantization_mode="int4-ready",
        backend="torch",
    )
