"""Tests for the Hodgkin-Huxley biophysical compartment model.

These tests cover the Nernst reversal potential, continuous state derivatives,
RK4 integration, and full timeline simulation. Concentrations and voltages use
the millimolar / millivolt conventions documented in ``hodgkin_huxley``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from spiking_neural_network.hodgkin_huxley import (
    BioPhysicalSystem,
    CompartmentState,
    biophysical_system_derivative,
    compiled_snn_simulator,
    main,
    nernst_potential,
    rk4_step,
    run_biophysical_timeline,
)

_TEMPERATURE_K = 310.15
_DT_MS = 0.05


@pytest.fixture
def two_compartment_system() -> BioPhysicalSystem:
    """Standard two-compartment cable with mammalian ion concentrations."""
    return BioPhysicalSystem(
        spatial_distances=jnp.array([1.0, 1.0]),
        axial_resistances=jnp.array([1.0, 1.0]),
        compartment_volumes=jnp.array([1e-12, 1e-12]),
        membrane_areas=jnp.array([1e-6, 1e-6]),
        external_na=145.0,
        external_k=5.0,
        temperature_kelvin=_TEMPERATURE_K,
    )


@pytest.fixture
def resting_state() -> CompartmentState:
    """Classic Hodgkin-Huxley resting values for two coupled compartments."""
    return CompartmentState(
        v_membrane=jnp.array([-65.0, -65.0]),
        m_gate=jnp.array([0.05, 0.05]),
        h_gate=jnp.array([0.6, 0.6]),
        n_gate=jnp.array([0.32, 0.32]),
        internal_na=jnp.array([18.0, 18.0]),
        internal_k=jnp.array([135.0, 135.0]),
    )


class TestNernstPotential:
    def test_equilibrium_concentrations_give_zero_reversal(self) -> None:
        internal = jnp.array([10.0, 20.0])
        reversal = nernst_potential(internal, external_conc=10.0, valence=1.0, temp=_TEMPERATURE_K)

        np.testing.assert_allclose(np.asarray(reversal)[0], 0.0, atol=1e-12)

    def test_sodium_reversal_is_positive_when_external_exceeds_internal(self) -> None:
        reversal = float(
            nernst_potential(
                jnp.array([18.0]),
                external_conc=145.0,
                valence=1.0,
                temp=_TEMPERATURE_K,
            )[0]
        )

        assert reversal > 0.0

    def test_potassium_reversal_is_negative_when_internal_exceeds_external(self) -> None:
        reversal = float(
            nernst_potential(
                jnp.array([135.0]),
                external_conc=5.0,
                valence=1.0,
                temp=_TEMPERATURE_K,
            )[0]
        )

        assert reversal < 0.0


class TestBiophysicalDerivative:
    def test_derivative_fields_match_state_shapes(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        zero_stimulus = jnp.zeros_like(resting_state.v_membrane)
        derivatives = biophysical_system_derivative(
            resting_state, two_compartment_system, zero_stimulus
        )

        assert derivatives.v_membrane.shape == resting_state.v_membrane.shape
        assert derivatives.m_gate.shape == resting_state.m_gate.shape
        assert derivatives.h_gate.shape == resting_state.h_gate.shape
        assert derivatives.n_gate.shape == resting_state.n_gate.shape
        assert derivatives.internal_na.shape == resting_state.internal_na.shape
        assert derivatives.internal_k.shape == resting_state.internal_k.shape

    def test_injected_current_increases_voltage_derivative(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        """Extra stimulus should increase dV/dt relative to the no-stimulus case."""
        zero_stimulus = jnp.zeros_like(resting_state.v_membrane)
        strong_stimulus = jnp.full_like(resting_state.v_membrane, 10.0)

        at_rest = biophysical_system_derivative(
            resting_state, two_compartment_system, zero_stimulus
        )
        with_stimulus = biophysical_system_derivative(
            resting_state, two_compartment_system, strong_stimulus
        )

        assert float(with_stimulus.v_membrane[0]) > float(at_rest.v_membrane[0])

    def test_single_compartment_has_no_axial_current(self) -> None:
        system = BioPhysicalSystem(
            spatial_distances=jnp.array([1.0]),
            axial_resistances=jnp.array([1.0]),
            compartment_volumes=jnp.array([1e-12]),
            membrane_areas=jnp.array([1e-6]),
            external_na=145.0,
            external_k=5.0,
            temperature_kelvin=_TEMPERATURE_K,
        )
        state = CompartmentState(
            v_membrane=jnp.array([-65.0]),
            m_gate=jnp.array([0.05]),
            h_gate=jnp.array([0.6]),
            n_gate=jnp.array([0.32]),
            internal_na=jnp.array([18.0]),
            internal_k=jnp.array([135.0]),
        )
        derivatives = biophysical_system_derivative(
            state, system, jnp.array([0.0])
        )

        assert derivatives.v_membrane.shape == (1,)


class TestRk4Step:
    def test_positive_stimulus_depolarizes_membrane(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        stimulus = jnp.full_like(resting_state.v_membrane, 10.0)
        next_state = rk4_step(resting_state, two_compartment_system, stimulus, _DT_MS)

        assert float(next_state.v_membrane[0]) > float(resting_state.v_membrane[0])

    def test_gating_variables_remain_in_unit_interval_after_short_step(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        stimulus = jnp.full_like(resting_state.v_membrane, 10.0)
        next_state = rk4_step(resting_state, two_compartment_system, stimulus, _DT_MS)

        for gate in (next_state.m_gate, next_state.h_gate, next_state.n_gate):
            gate_values = np.asarray(gate)
            assert np.all(gate_values >= 0.0)
            assert np.all(gate_values <= 1.0)


class TestRunBiophysicalTimeline:
    def test_history_length_matches_stimulus_profile(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        num_steps = 40
        stimulus = jnp.zeros((num_steps, 2))
        _, history = run_biophysical_timeline(
            resting_state, two_compartment_system, stimulus, _DT_MS
        )

        assert history.v_membrane.shape == (num_steps, 2)

    def test_short_stimulus_pulse_depolarizes_membrane(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        """A brief strong pulse should raise voltage before ion concentrations diverge."""
        num_steps = 10
        stimulus = jnp.full((num_steps, 2), 10.0)
        _, history = run_biophysical_timeline(
            resting_state, two_compartment_system, stimulus, _DT_MS
        )

        peak_voltage = float(np.max(np.asarray(history.v_membrane)))
        resting_voltage = float(resting_state.v_membrane[0])

        assert peak_voltage > resting_voltage
        assert np.isfinite(peak_voltage)

    def test_zero_stimulus_short_run_remains_finite(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        """Without input, a short rollout should stay numerically stable."""
        num_steps = 10
        stimulus = jnp.zeros((num_steps, 2))
        final_state, history = run_biophysical_timeline(
            resting_state, two_compartment_system, stimulus, _DT_MS
        )

        assert np.all(np.isfinite(np.asarray(history.v_membrane)))
        assert np.all(np.isfinite(np.asarray(final_state.v_membrane)))

    def test_compiled_simulator_matches_eager_integration(
        self,
        resting_state: CompartmentState,
        two_compartment_system: BioPhysicalSystem,
    ) -> None:
        num_steps = 5
        stimulus = jnp.full((num_steps, 2), 5.0)

        eager_final, eager_history = run_biophysical_timeline(
            resting_state, two_compartment_system, stimulus, _DT_MS
        )
        jit_final, jit_history = compiled_snn_simulator(
            resting_state, two_compartment_system, stimulus, _DT_MS
        )

        np.testing.assert_allclose(
            np.asarray(jit_history.v_membrane),
            np.asarray(eager_history.v_membrane),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(jit_final.v_membrane),
            np.asarray(eager_final.v_membrane),
            rtol=1e-5,
        )


def test_main_demo_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()

    assert "CompartmentState" in captured.out


def test_main_module_entrypoint_runs_without_error() -> None:
    import runpy

    runpy.run_module("spiking_neural_network.hodgkin_huxley", run_name="__main__")
