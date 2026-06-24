"""Tests for epoch context and learning-rate schedules."""

from __future__ import annotations

import pytest

from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import AdaLiConfig
from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import (
    EpochContext,
    cosine_learning_rate,
    linear_learning_rate,
)


class TestLearningRateSchedules:
    def test_linear_learning_rate_decays_over_epochs(self) -> None:
        schedule = linear_learning_rate(0.1, 0.01)
        assert schedule(EpochContext(1, 5)) == pytest.approx(0.1)
        assert schedule(EpochContext(5, 5)) == pytest.approx(0.01)
        assert schedule(EpochContext(3, 5)) == pytest.approx(0.055)

    def test_cosine_learning_rate_decays_over_epochs(self) -> None:
        schedule = cosine_learning_rate(0.1, 0.01)
        assert schedule(EpochContext(1, 5)) == pytest.approx(0.1)
        assert schedule(EpochContext(5, 5)) == pytest.approx(0.01)
        assert schedule(EpochContext(3, 5)) == pytest.approx(0.055)

    def test_default_learning_rate_is_cosine_schedule(self) -> None:
        model = AdaLi(AdaLiConfig())

        assert model.learning_rate_at(EpochContext(1, 10)) == pytest.approx(0.01)
        assert model.learning_rate_at(EpochContext(10, 10)) == pytest.approx(0.001)

    def test_epoch_context_progress(self) -> None:
        assert EpochContext(1, 5).progress == 0.0
        assert EpochContext(5, 5).progress == 1.0
        assert EpochContext(3, 5).progress == pytest.approx(0.5)

    def test_linear_schedule_single_epoch(self) -> None:
        schedule = linear_learning_rate(0.1, 0.01)
        assert schedule(EpochContext(1, 1)) == pytest.approx(0.1)

    def test_cosine_schedule_single_epoch(self) -> None:
        schedule = cosine_learning_rate(0.1, 0.01)
        assert schedule(EpochContext(1, 1)) == pytest.approx(0.1)

    def test_linear_learning_rate_rejects_non_positive_initial(self) -> None:
        with pytest.raises(ParameterError, match="initial learning rate must be positive"):
            linear_learning_rate(0.0, 0.01)

    def test_cosine_learning_rate_rejects_non_positive_final(self) -> None:
        with pytest.raises(ParameterError, match="final learning rate must be positive"):
            cosine_learning_rate(0.1, 0.0)

    def test_cosine_learning_rate_rejects_non_positive_initial(self) -> None:
        with pytest.raises(ParameterError, match="initial learning rate must be positive"):
            cosine_learning_rate(0.0, 0.01)

    def test_linear_learning_rate_rejects_non_positive_final(self) -> None:
        with pytest.raises(ParameterError, match="final learning rate must be positive"):
            linear_learning_rate(0.1, 0.0)
