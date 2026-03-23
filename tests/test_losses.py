import torch

from aerorl import build_text_token_mask, masked_cross_entropy_loss


def test_build_text_token_mask_excludes_vision_and_ignore() -> None:
    labels = torch.tensor([[1, 2, -100, 4]])
    vision_mask = torch.tensor([[True, False, False, True]])

    mask = build_text_token_mask(vision_mask=vision_mask, labels=labels)
    expected = torch.tensor([[False, True, False, False]])

    assert torch.equal(mask, expected)


def test_masked_cross_entropy_loss_runs() -> None:
    logits = torch.randn(1, 4, 8)
    labels = torch.tensor([[1, 2, 3, 4]])
    vision_mask = torch.tensor([[True, False, False, True]])

    loss = masked_cross_entropy_loss(logits=logits, labels=labels, vision_mask=vision_mask)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
