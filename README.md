# Spiking-Neural-Network

Convert grayscale images into Poisson spike trains for spiking neural network experiments.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --group dev
```

## Run

```bash
uv run python scripts/preprocess.py
```

## Test, format, and typecheck

```bash
uv run --group dev pytest
uv run --group dev black --check .
uv run --group dev basedpyright
```

## Layout

```
src/spiking_neural_network/
  config.py      PreprocessConfig and EncodingConfig
  images.py      Load, resize, normalize images
  encoding.py    Poisson spike encoding
  plotting.py    Spike/rate visualization
  validation.py  Relative error metric

scripts/preprocess.py   End-to-end demo
tests/                  Pytest suite
```

## Pipeline

1. Load grayscale image
2. Resize (default 32×32)
3. Normalize pixel values to [0, 1]
4. Sample Poisson spike counts over `t_steps`
5. Compare expected vs observed never-spike pixel count
6. Plot rates, raster, and population activity
