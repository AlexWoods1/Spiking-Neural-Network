"""Epoch context, learning-rate schedules, and surrogate boundary state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, pi

from spiking_neural_network.exceptions import ParameterError


@dataclass(frozen=True)
class EpochContext:
    epoch: int
    total_epochs: int

    @property
    def progress(self) -> float:
        if self.total_epochs == 1 or self.epoch <= 1:
            return 0.0
        if self.epoch >= self.total_epochs:
            return 1.0
        return (self.epoch - 1) / (self.total_epochs - 1)


LearningRateSchedule = Callable[[EpochContext], float]


def _linear_schedule(
    ctx: EpochContext,
    initial: float,
    final: float,
) -> float:
    if ctx.total_epochs == 1 or ctx.epoch <= 1:
        return initial
    if ctx.epoch >= ctx.total_epochs:
        return final
    return (final - initial) / (ctx.total_epochs - 1) * (ctx.epoch - 1) + initial


def linear_learning_rate(initial: float, final: float) -> LearningRateSchedule:
    """Build a learning-rate schedule that linearly interpolates by epoch."""
    if initial <= 0:
        raise ParameterError("initial learning rate must be positive")
    if final <= 0:
        raise ParameterError("final learning rate must be positive")

    def schedule(ctx: EpochContext) -> float:
        return _linear_schedule(ctx, initial, final)

    return schedule


def cosine_learning_rate(initial: float, final: float) -> LearningRateSchedule:
    """Build a cosine-annealing schedule from ``initial`` to ``final`` by epoch."""
    if initial <= 0:
        raise ParameterError("initial learning rate must be positive")
    if final <= 0:
        raise ParameterError("final learning rate must be positive")

    def schedule(ctx: EpochContext) -> float:
        if ctx.total_epochs == 1:
            return initial
        progress = (ctx.epoch - 1) / (ctx.total_epochs - 1)
        cosine = 0.5 * (1.0 + cos(pi * progress))
        return final + (initial - final) * cosine

    return schedule


DEFAULT_LEARNING_RATE_INITIAL = 0.01
DEFAULT_LEARNING_RATE_FINAL = 0.001
DEFAULT_LEARNING_RATE = cosine_learning_rate(
    DEFAULT_LEARNING_RATE_INITIAL,
    DEFAULT_LEARNING_RATE_FINAL,
)


def _validate_learning_rate(learning_rate: float | LearningRateSchedule) -> None:
    if isinstance(learning_rate, (int, float)):
        if learning_rate <= 0:
            raise ParameterError("learning_rate must be positive")
        return
    if callable(learning_rate):
        probe = learning_rate(EpochContext(1, 1))
        if not isinstance(probe, (int, float)) or probe <= 0:
            raise ParameterError("learning_rate schedule must return a positive number")
        return
    raise ParameterError("learning_rate must be a positive number or callable(ctx: EpochContext)")


@dataclass(frozen=True)
class BoundaryState:
    """Adaptive surrogate window ``(V^-, V^+)`` for one training epoch."""

    v_minus: float
    v_plus: float


@dataclass(frozen=True)
class EpochTrainingState:
    ctx: EpochContext
    learning_rate: float
