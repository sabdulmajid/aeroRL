from __future__ import annotations

import torch
import torch.nn.functional as F


def build_text_token_mask(vision_mask: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if vision_mask.shape != labels.shape:
        raise ValueError("vision_mask and labels must have identical shape")
    if vision_mask.dtype != torch.bool:
        vision_mask = vision_mask.to(torch.bool)
    valid_labels = labels.ne(ignore_index)
    return (~vision_mask) & valid_labels


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vision_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits batch/seq dimensions must match labels")

    token_mask = build_text_token_mask(vision_mask=vision_mask, labels=labels, ignore_index=ignore_index)
    safe_labels = labels.masked_fill(~token_mask, ignore_index)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        safe_labels.reshape(-1),
        ignore_index=ignore_index,
    )
