from dataclasses import dataclass


@dataclass(slots=True)
class AeroRLConfig:
    zero_copy_kv: bool = True
    mask_vision_tokens: bool = True
    quant_ref_bits: int = 8
    trainer_backend: str = "auto"

    def validate(self) -> None:
        if self.quant_ref_bits not in {4, 8, 16}:
            raise ValueError("quant_ref_bits must be one of {4, 8, 16}")
        if self.trainer_backend not in {"auto", "trl", "verl"}:
            raise ValueError("trainer_backend must be one of {'auto', 'trl', 'verl'}")
