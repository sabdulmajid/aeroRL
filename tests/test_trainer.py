import torch

from aerorl import AeroRLConfig, AeroRLTrainer


def test_trainer_lifecycle_and_step() -> None:
    trainer = AeroRLTrainer(AeroRLConfig(mask_vision_tokens=True))

    start = trainer.on_train_start()
    assert start["status"] == "started"

    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    vision_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])

    result = trainer.train_step(logits=logits, labels=labels, vision_mask=vision_mask, optimizer=None)
    assert result["step"] == 1
    assert result["loss"] >= 0.0

    end = trainer.on_train_end()
    assert end["status"] == "finished"
    assert end["steps"] == 1
