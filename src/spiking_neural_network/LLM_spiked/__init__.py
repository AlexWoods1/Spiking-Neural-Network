"""Spiking Shakespeare LM: causal attention + Spiking LSTM (JAX)."""

from spiking_neural_network.LLM_spiked.config import (
    Config,
    DataConfig,
    ModelConfig,
    PathsConfig,
    TrainConfig,
    load_config,
)
from spiking_neural_network.LLM_spiked.data import CharDataset, CharTokenizer
from spiking_neural_network.LLM_spiked.generate import generate, load_checkpoint
from spiking_neural_network.LLM_spiked.model import (
    count_parameters,
    forward,
    init_params,
    loss_fn,
)
from spiking_neural_network.LLM_spiked.spikes import spike
from spiking_neural_network.LLM_spiked.train import get_lr, train

__all__ = [
    "CharDataset",
    "CharTokenizer",
    "Config",
    "DataConfig",
    "ModelConfig",
    "PathsConfig",
    "TrainConfig",
    "count_parameters",
    "forward",
    "generate",
    "get_lr",
    "init_params",
    "load_checkpoint",
    "load_config",
    "loss_fn",
    "spike",
    "train",
]
