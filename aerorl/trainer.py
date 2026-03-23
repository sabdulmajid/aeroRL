from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .adapters import resolve_trainer_backend
from .config import AeroRLConfig
from .losses import masked_cross_entropy_loss


@dataclass(slots=True)
class TrainerState:
    step: int = 0
    started: bool = False
    finished: bool = False


class AeroRLTrainer:
    def __init__(self, config: AeroRLConfig):
        config.validate()
        self.config = config
        self.adapter = resolve_trainer_backend(config.trainer_backend)
        self.state = TrainerState()

    def on_train_start(self) -> dict[str, Any]:
        self.state.started = True
        self.state.finished = False
        return {
            "status": "started",
            "backend": self.adapter.backend,
            "backend_available": self.adapter.available,
        }

    def compute_masked_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.mask_vision_tokens:
            return masked_cross_entropy_loss(logits=logits, labels=labels, vision_mask=vision_mask)

        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

    def train_step(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vision_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        if not self.state.started:
            self.on_train_start()

        loss = self.compute_masked_loss(logits=logits, labels=labels, vision_mask=vision_mask)
        loss.backward()

        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.state.step += 1
        return {
            "step": self.state.step,
            "loss": float(loss.detach().cpu().item()),
            "backend": self.adapter.backend,
        }

    def on_train_end(self) -> dict[str, Any]:
        self.state.finished = True
        return {
            "status": "finished",
            "steps": self.state.step,
            "backend": self.adapter.backend,
        }
