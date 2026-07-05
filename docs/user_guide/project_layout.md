# Project Layout

```
src/spiking_neural_network/
  encoding.py, images.py, lif.py, network.py   Spike encoding and LIF layers
  datasets.py, data_module.py                  MNIST loading and batches
  adali/                                       AdaLi model (JAX only)
  pipeline.py                                  Training factories and public API
  trainer.py, evaluation.py, training_logs.py  Train loop, metrics, exports

scripts/preprocess.py   Encoding demo
scripts/train.py        MNIST AdaLi training CLI
tests/                  Pytest suite
docs/                   Sphinx user guide and API reference
```
