import spiking_neural_network as snn


def test_public_exports_are_importable() -> None:
    assert snn.EncodingConfig is not None
    assert snn.PreprocessConfig is not None
    assert snn.SpikeEncoding is not None
    assert snn.relative_error(1.0, 1.0) == 0.0
