import spiking_neural_network as snn


def test_public_exports_are_importable() -> None:
    assert snn.EncodingConfig is not None
    assert snn.PreprocessConfig is not None
    assert snn.SpikeEncoding is not None
    assert snn.relative_error(1.0, 1.0) == 0.0


def test_lazy_exports_resolve_on_first_access() -> None:
    assert "plot_spikes" not in snn.__dict__
    assert callable(snn.plot_spikes)
    assert "plot_spikes" in snn.__dict__
    assert "load_grayscale" not in snn.__dict__ or callable(snn.load_grayscale)
