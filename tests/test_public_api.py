from aerorl import AeroRLConfig, wrap_vlm_for_rl


def test_public_api_smoke() -> None:
    config = AeroRLConfig(quant_ref_bits=8)
    model, ref_model = wrap_vlm_for_rl("demo/model", config)

    assert model["aerorl"]["mask_vision_tokens"] is True
    assert "trainer" in model
    assert ref_model["precision"] == "int8"
    assert ref_model["quantization_mode"].startswith("int8-")
