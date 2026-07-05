# Quickstart

## Poisson encoding demo

```bash
uv run python scripts/preprocess.py
```

## Train AdaLi on MNIST (JAX backend)

```bash
uv run python scripts/train.py --epochs 5 --fast
```

Training outputs land in `outputs/` (`history.csv`, `run.json`, `class_summary.csv`).
MNIST downloads to `data/mnist/` on first run.

## Test, format, and typecheck

```bash
uv run --group dev pytest
uv run --group dev black --check .
uv run --group dev basedpyright
```

## Build documentation

```bash
uv run --group dev sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to view the site.
