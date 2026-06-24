"""Hodgkin-Huxley biophysical neuron dynamics with JAX integration.

This module models membrane voltage, ion-channel gating, and intracellular ion
concentrations as a continuous physical system. Action potentials emerge from the
coupled differential equations rather than from explicit spike thresholds.

Units (unless noted otherwise):
    Time: milliseconds (ms)
    Voltage: millivolts (mV)
    Current density: microamperes per square centimeter (uA/cm^2)
    Concentration: millimolar (mM)
    Temperature: kelvin (K)
    Length: micrometers (um)
    Resistance: megaohms (MOhm)
    Volume: liters (L)
    Area: square centimeters (cm^2)
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


class CompartmentState(NamedTuple):
    """Dynamic state of one or more electrically coupled compartments.

    Attributes:
        v_membrane: Transmembrane voltage per compartment (mV).
        m_gate: Sodium activation gate (dimensionless, in [0, 1]).
        h_gate: Sodium inactivation gate (dimensionless, in [0, 1]).
        n_gate: Potassium activation gate (dimensionless, in [0, 1]).
        internal_na: Intracellular sodium concentration per compartment (mM).
        internal_k: Intracellular potassium concentration per compartment (mM).
    """

    v_membrane: jnp.ndarray
    m_gate: jnp.ndarray
    h_gate: jnp.ndarray
    n_gate: jnp.ndarray
    internal_na: jnp.ndarray
    internal_k: jnp.ndarray


class BioPhysicalSystem(NamedTuple):
    """Static physical parameters of the simulated tissue.

    Attributes:
        spatial_distances: Distance between adjacent compartments (um).
        axial_resistances: Axial resistance linking compartments (MOhm).
        compartment_volumes: Compartment volume (L).
        membrane_areas: Membrane surface area (cm^2).
        external_na: Extracellular sodium concentration (mM).
        external_k: Extracellular potassium concentration (mM).
        temperature_kelvin: Absolute temperature (K).
    """

    spatial_distances: jnp.ndarray
    axial_resistances: jnp.ndarray
    compartment_volumes: jnp.ndarray
    membrane_areas: jnp.ndarray
    external_na: float
    external_k: float
    temperature_kelvin: float


# RT/F in mV/K for the Nernst equation (R and F in SI units, scaled to mV).
_NERNST_MV_PER_K = 0.0861733319277716


def nernst_potential(
    internal_conc: jnp.ndarray,
    external_conc: float,
    valence: float,
    temp: float,
) -> jnp.ndarray:
    """Compute the Nernst equilibrium potential for one ion species.

    Args:
        internal_conc: Intracellular concentration (mM).
        external_conc: Extracellular concentration (mM).
        valence: Signed ion valence (for example, +1 for Na+, +1 for K+).
        temp: Absolute temperature (K).

    Returns:
        Reversal potential in millivolts (mV).
    """
    return _NERNST_MV_PER_K * temp * jnp.log(external_conc / internal_conc) / valence


def biophysical_system_derivative(
    state: CompartmentState,
    system: BioPhysicalSystem,
    injected_current: jnp.ndarray,
) -> CompartmentState:
    """Evaluate dX/dt for the coupled Hodgkin-Huxley compartment system.

    The returned ``CompartmentState`` holds time derivatives, not absolute state
    values. Ionic currents, gating kinetics, axial cable flow, and concentration
    flux are all computed from the current physical state.

    Args:
        state: Current compartment state.
        system: Static biophysical parameters.
        injected_current: External stimulus current density per compartment
            (uA/cm^2).

    Returns:
        Time derivatives for each field in ``CompartmentState`` (units per ms).
    """
    v_membrane = state.v_membrane
    temp = system.temperature_kelvin

    e_na = nernst_potential(state.internal_na, system.external_na, 1.0, temp)
    e_k = nernst_potential(state.internal_k, system.external_k, 1.0, temp)
    e_l = -54.387

    alpha_m = 0.1 * (v_membrane + 40) / (1 - jnp.exp(-(v_membrane + 40) / 10))
    beta_m = 4.0 * jnp.exp(-(v_membrane + 65) / 18)
    dm_dt = alpha_m * (1 - state.m_gate) - beta_m * state.m_gate

    alpha_h = 0.07 * jnp.exp(-(v_membrane + 65) / 20)
    beta_h = 1 / (1 + jnp.exp(-(v_membrane + 35) / 10))
    dh_dt = alpha_h * (1 - state.h_gate) - beta_h * state.h_gate

    alpha_n = 0.01 * (v_membrane + 55) / (1 - jnp.exp(-(v_membrane + 55) / 10))
    beta_n = 0.125 * jnp.exp(-(v_membrane + 65) / 80)
    dn_dt = alpha_n * (1 - state.n_gate) - beta_n * state.n_gate

    g_na = 120.0 * state.m_gate**3 * state.h_gate
    g_k = 36.0 * state.n_gate**4
    g_l = 0.3

    i_na = g_na * (v_membrane - e_na)
    i_k = g_k * (v_membrane - e_k)
    i_l = g_l * (v_membrane - e_l)
    i_ionic = i_na + i_k + i_l

    if v_membrane.shape[0] >= 2:
        grad_v = jnp.asarray(jnp.gradient(v_membrane, axis=0))
        i_spatial = grad_v / system.axial_resistances
    else:
        i_spatial = jnp.zeros_like(v_membrane)

    c_m = 1.0
    dv_dt = (-i_ionic + i_spatial + injected_current) / c_m

    faraday = 96485.3329
    abs_current_na = i_na * system.membrane_areas
    abs_current_k = i_k * system.membrane_areas
    flux_na = abs_current_na / faraday
    flux_k = abs_current_k / faraday
    d_na_dt = -flux_na / system.compartment_volumes
    d_k_dt = -flux_k / system.compartment_volumes

    return CompartmentState(
        v_membrane=dv_dt,
        m_gate=dm_dt,
        h_gate=dh_dt,
        n_gate=dn_dt,
        internal_na=d_na_dt,
        internal_k=d_k_dt,
    )


def rk4_step(
    state: CompartmentState,
    system: BioPhysicalSystem,
    injected_current: jnp.ndarray,
    dt: float,
) -> CompartmentState:
    """Advance the biophysical state by one explicit RK4 step.

    Args:
        state: Current compartment state.
        system: Static biophysical parameters.
        injected_current: External stimulus current density (uA/cm^2).
        dt: Integration step size (ms).

    Returns:
        Updated compartment state after integrating forward by ``dt``.
    """
    k1 = biophysical_system_derivative(state, system, injected_current)
    half_dt = dt / 2

    state_k2 = CompartmentState(
        v_membrane=state.v_membrane + k1.v_membrane * half_dt,
        m_gate=state.m_gate + k1.m_gate * half_dt,
        h_gate=state.h_gate + k1.h_gate * half_dt,
        n_gate=state.n_gate + k1.n_gate * half_dt,
        internal_na=state.internal_na + k1.internal_na * half_dt,
        internal_k=state.internal_k + k1.internal_k * half_dt,
    )
    k2 = biophysical_system_derivative(state_k2, system, injected_current)

    state_k3 = CompartmentState(
        v_membrane=state.v_membrane + k2.v_membrane * half_dt,
        m_gate=state.m_gate + k2.m_gate * half_dt,
        h_gate=state.h_gate + k2.h_gate * half_dt,
        n_gate=state.n_gate + k2.n_gate * half_dt,
        internal_na=state.internal_na + k2.internal_na * half_dt,
        internal_k=state.internal_k + k2.internal_k * half_dt,
    )
    k3 = biophysical_system_derivative(state_k3, system, injected_current)

    state_k4 = CompartmentState(
        v_membrane=state.v_membrane + k3.v_membrane * dt,
        m_gate=state.m_gate + k3.m_gate * dt,
        h_gate=state.h_gate + k3.h_gate * dt,
        n_gate=state.n_gate + k3.n_gate * dt,
        internal_na=state.internal_na + k3.internal_na * dt,
        internal_k=state.internal_k + k3.internal_k * dt,
    )
    k4 = biophysical_system_derivative(state_k4, system, injected_current)
    sixth_dt = dt / 6.0

    return CompartmentState(
        v_membrane=state.v_membrane
        + sixth_dt
        * (k1.v_membrane + 2 * k2.v_membrane + 2 * k3.v_membrane + k4.v_membrane),
        m_gate=state.m_gate
        + sixth_dt * (k1.m_gate + 2 * k2.m_gate + 2 * k3.m_gate + k4.m_gate),
        h_gate=state.h_gate
        + sixth_dt * (k1.h_gate + 2 * k2.h_gate + 2 * k3.h_gate + k4.h_gate),
        n_gate=state.n_gate
        + sixth_dt * (k1.n_gate + 2 * k2.n_gate + 2 * k3.n_gate + k4.n_gate),
        internal_na=state.internal_na
        + sixth_dt
        * (k1.internal_na + 2 * k2.internal_na + 2 * k3.internal_na + k4.internal_na),
        internal_k=state.internal_k
        + sixth_dt
        * (k1.internal_k + 2 * k2.internal_k + 2 * k3.internal_k + k4.internal_k),
    )


def run_biophysical_timeline(
    initial_state: CompartmentState,
    system_config: BioPhysicalSystem,
    injected_current_profile: jnp.ndarray,
    dt: float,
) -> Tuple[CompartmentState, CompartmentState]:
    """Integrate the model across a sequence of stimulus values.

    Args:
        initial_state: State at t = 0.
        system_config: Static biophysical parameters.
        injected_current_profile: Stimulus current density at each step
            (uA/cm^2), one value (or vector) per integration step.
        dt: Integration step size (ms).

    Returns:
        A tuple ``(final_state, history)`` where ``final_state`` is the state
        after the last step and ``history`` stacks the state recorded at every
        step (including the final one).
    """

    def scan_body(
        carry_state: CompartmentState, current_stimulus: jnp.ndarray
    ) -> tuple[CompartmentState, CompartmentState]:
        next_state = rk4_step(carry_state, system_config, current_stimulus, dt)
        return next_state, next_state

    final_state, history = jax.lax.scan(
        scan_body, initial_state, injected_current_profile
    )
    return final_state, history


compiled_snn_simulator = jax.jit(run_biophysical_timeline)


def main() -> None:
    """Run a short two-compartment Hodgkin-Huxley demonstration."""
    total_time_ms = 50.0
    dt = 0.05
    num_steps = int(total_time_ms / dt)
    stimulus = jnp.zeros(num_steps)
    stimulus = stimulus.at[int(5.0 / dt) :].set(10.0)

    biological_environment = BioPhysicalSystem(
        spatial_distances=jnp.array([1.0, 1.0]),
        axial_resistances=jnp.array([1.0, 1.0]),
        compartment_volumes=jnp.array([1e-12, 1e-12]),
        membrane_areas=jnp.array([1e-6, 1e-6]),
        external_na=145.0,
        external_k=5.0,
        temperature_kelvin=310.15,
    )

    initial_state = CompartmentState(
        v_membrane=jnp.array([-65.0, -65.0]),
        m_gate=jnp.array([0.05, 0.05]),
        h_gate=jnp.array([0.6, 0.6]),
        n_gate=jnp.array([0.32, 0.32]),
        internal_na=jnp.array([18.0, 18.0]),
        internal_k=jnp.array([135.0, 135.0]),
    )
    final_state, state_history = compiled_snn_simulator(
        initial_state, biological_environment, stimulus, dt
    )
    print(final_state)
    print(state_history)


if __name__ == "__main__":
    main()
