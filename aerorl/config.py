from dataclasses import dataclass


@dataclass(slots=True)
class AeroRLConfig:
    zero_copy_kv: bool = True
    mask_vision_tokens: bool = True
    quant_ref_bits: int = 8

    def validate(self) -> None:
        if self.quant_ref_bits not in {4, 8, 16}:
            raise ValueError("quant_ref_bits must be one of {4, 8, 16}")
