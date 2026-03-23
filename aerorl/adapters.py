from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrainerAdapter:
    backend: str
    available: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "reason": self.reason,
        }


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_trainer_backend(preferred: str = "auto") -> TrainerAdapter:
    if preferred == "trl":
        available = _has_module("trl")
        reason = "trl installed" if available else "trl not installed"
        return TrainerAdapter(backend="trl", available=available, reason=reason)

    if preferred == "verl":
        available = _has_module("verl")
        reason = "verl installed" if available else "verl not installed"
        return TrainerAdapter(backend="verl", available=available, reason=reason)

    if _has_module("trl"):
        return TrainerAdapter(backend="trl", available=True, reason="auto-selected trl")

    if _has_module("verl"):
        return TrainerAdapter(backend="verl", available=True, reason="auto-selected verl")

    return TrainerAdapter(
        backend="none",
        available=False,
        reason="neither trl nor verl is installed; running in scaffold mode",
    )
