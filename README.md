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

Preprocess the sample image, print a validation metric, and show a spike plot:

```bash
uv run python scripts/preprocess.py
```

## Test & format

```bash
uv run --group dev pytest
uv run --group dev black --check .
```

## Layout

```
src/spiking_neural_network/
  config.py      Encoding settings (t_steps, seed)
  images.py      Load, resize, normalize images
  encoding.py    Poisson spike encoding
  plotting.py    Spike/rate visualization
  validation.py  Relative error metric

scripts/preprocess.py   End-to-end demo
tests/                  Pytest suite
```

## Pipeline

1. Load grayscale image
2. Resize (default 32×32 in the demo script)
3. Normalize pixel values to [0, 1]
4. Sample Poisson spike counts over `t_steps`
5. Compare expected vs observed never-spike pixel count
