# Spiking-Neural-Network

neuron think strong.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --group dev
```

## Run

Poisson encoding demo:

```bash
uv run python scripts/preprocess.py
```

Train AdaLi on MNIST (JAX backend):

```bash
uv run python scripts/train.py --epochs 5 --fast
```

Outputs land in `outputs/` (`history.csv`, `run.json`, `class_summary.csv`). MNIST downloads to `data/mnist/` on first run.

## Test, format, and typecheck

```bash
uv run --group dev pytest
uv run --group dev black --check .
uv run --group dev basedpyright
```

## Layout

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
```
