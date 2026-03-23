from aerorl import create_quantized_reference_runtime, resolve_trainer_backend


def test_auto_adapter_returns_structured_result() -> None:
    adapter = resolve_trainer_backend("auto")

    assert adapter.backend in {"trl", "verl", "none"}
    assert isinstance(adapter.available, bool)
    assert isinstance(adapter.reason, str)


def test_quantized_runtime_modes() -> None:
    ref8 = create_quantized_reference_runtime("demo/model", quant_bits=8)
    ref4 = create_quantized_reference_runtime("demo/model", quant_bits=4)
    ref16 = create_quantized_reference_runtime("demo/model", quant_bits=16)

    assert ref8.quantization_mode == "int8-ready"
    assert ref4.quantization_mode == "int4-ready"
    assert ref16.quantization_mode == "fp16-reference"
