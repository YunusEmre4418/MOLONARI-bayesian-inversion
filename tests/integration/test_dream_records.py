"""Integration tests of what DREAM records (InversionColumn.compute_mcmc with several chains) on the demo column:
the states, their energies, the acceptance rates, the quantiles and the sigma2 proposals."""

import numpy as np
import pytest

from bayesian_inversion import Prior, inversion_column
from bayesian_inversion.config import SIGMA2_MAX_T

pytestmark = pytest.mark.slow  # every test of this file needs an MCMC run

# Demo notebook settings
N_CHAIN = 10
N_ITER = 35
N_BURN_IN_MAX = 20

# Shorter runs for the sigma2 tests (at least 2 * delta + 1 = 7 chains for DREAM)
SHORT_N_ITER = 10
SHORT_N_BURN_IN_MAX = 5
SIGMA2_MIN_LINEAR = 0.1  # ratio 10 to SIGMA2_MAX_T, below the log threshold: sigma2 stays on a linear scale

ENERGY_RTOL = 1e-12      # the replay solves the same float64 parameters again
QUANTILE_RTOL = 1e-6


@pytest.fixture(scope="module")
def dream_column(run_demo_mcmc):
    """One DREAM run with the demo settings, shared by the tests of this file."""
    return run_demo_mcmc(nb_iter=N_ITER, nb_chain=N_CHAIN, nitmaxburning=N_BURN_IN_MAX)


def state_after(states, iteration, chain):
    """State of a chain after a main loop iteration: the history starts with one state per chain
    at the end of burn-in, then adds one state per chain and iteration."""
    return states[N_CHAIN * (iteration + 1) + chain]


def test_dream_records_one_state_per_chain_and_iteration(dream_column):
    """End of burn-in, then each iteration: N_CHAIN * (N_ITER + 1) states."""
    assert len(dream_column._states) == N_CHAIN * (N_ITER + 1)


def test_every_dream_state_energy_matches_its_parameters(dream_column, replay):
    """Each recorded energy is the energy of the parameters recorded with it, for every state."""
    unique_states = {id(state): state for state in dream_column._states}.values()  # rejections repeat a state
    inconsistent = [state.energy for state in unique_states
                    if replay(state)[0] != pytest.approx(state.energy, rel=ENERGY_RTOL)]
    assert not inconsistent, f"{len(inconsistent)} of {len(unique_states)} states, energies {inconsistent[:5]}"


def test_dream_acceptance_rate_is_accepted_over_proposed(dream_column):
    """The rate of each chain is the fraction of main loop iterations where its state changed."""
    states = dream_column._states
    for chain in range(N_CHAIN):
        changes = [not np.array_equal(state_after(states, i - 1, chain).layers, state_after(states, i, chain).layers)
                   for i in range(N_ITER)]
        assert dream_column.get_all_acceptance_ratio()[chain] == pytest.approx(np.mean(changes))


def test_dream_quantiles_use_only_accepted_states(dream_column, replay):
    """The quantiles are those of the fields of the accepted states, all chains pooled.
    With these settings every iteration is kept for the quantiles."""
    accepted = [state_after(dream_column._states, i, chain) for i in range(N_ITER) for chain in range(N_CHAIN)]
    fields = [replay(state)[1:] for state in accepted]
    temperatures = np.array([t for t, _ in fields], np.float32)
    flows = np.array([f for _, f in fields], np.float32)

    for quantile in dream_column.get_quantiles():
        np.testing.assert_allclose(dream_column.get_temperatures_quantile(quantile),
                                   np.quantile(temperatures, quantile, axis=0), rtol=QUANTILE_RTOL)
        np.testing.assert_allclose(dream_column.get_flows_quantile(quantile),
                                   np.quantile(flows, quantile, axis=0), rtol=QUANTILE_RTOL)


def test_dream_sigma2_proposals_start_from_current_state(run_demo_mcmc, monkeypatch):
    """In the estimesigma mode, each main loop proposal of sigma2 starts from the sigma2 of the current
    state of its chain. The sigma2 range is made linear, so that the MCMC and physical spaces coincide."""
    monkeypatch.setattr(inversion_column, "SIGMA2_MIN_T", SIGMA2_MIN_LINEAR)
    sigma2_range = (SIGMA2_MIN_LINEAR, SIGMA2_MAX_T)
    inputs = []
    original_perturb = Prior.perturb

    def recording_perturb(prior, mcmc_value):
        if not prior.is_fixed and tuple(prior.user_range) == sigma2_range:
            inputs.append(float(mcmc_value))
        return original_perturb(prior, mcmc_value)

    monkeypatch.setattr(Prior, "perturb", recording_perturb)
    column = run_demo_mcmc(nb_iter=SHORT_N_ITER, nb_chain=N_CHAIN, nitmaxburning=SHORT_N_BURN_IN_MAX,
                           typealgo="estimesigma")

    main_loop_inputs = inputs[-SHORT_N_ITER * N_CHAIN:]  # one proposal per chain and iteration, chains in order
    current = [float(column._states[N_CHAIN * i + chain].sigma2_temp)
               for i in range(SHORT_N_ITER) for chain in range(N_CHAIN)]
    assert main_loop_inputs == current


def test_dream_estimesigma_records_positive_sigma2(run_demo_mcmc):
    """With the default sigma2 range, the recorded sigma2 are variances, so they are positive."""
    column = run_demo_mcmc(nb_iter=SHORT_N_ITER, nb_chain=N_CHAIN, nitmaxburning=SHORT_N_BURN_IN_MAX,
                           typealgo="estimesigma")
    assert all(state.sigma2_temp > 0 for state in column._states)
