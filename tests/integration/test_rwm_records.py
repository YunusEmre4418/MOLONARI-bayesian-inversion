"""Integration tests of what the random walk Metropolis records (InversionColumn.compute_mcmc with one chain) on the
demo column: the states and the space they are stored in, the parameter getters, the acceptance rate
and the quantiles."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow  # every test of this file needs an MCMC run

# Settings of the reference RWM run
N_ITER = 100
N_BURN_IN_MAX = 20
ONE_CHAIN = 1

ENERGY_RTOL = 1e-6
QUANTILE_RTOL = 1e-6


@pytest.fixture(scope="module")
def rwm_column(run_demo_mcmc):
    """One RWM run, shared by the tests of this file."""
    return run_demo_mcmc(nb_iter=N_ITER, nb_chain=ONE_CHAIN, nitmaxburning=N_BURN_IN_MAX)


def state_after(states, iteration):
    """State after a main loop iteration: the history starts with the best initial state."""
    return states[iteration + 1]


def test_rwm_records_initial_state_and_one_state_per_iteration(rwm_column):
    assert len(rwm_column._states) == N_ITER + 1


def test_every_rwm_state_energy_matches_its_parameters_in_mcmc_space(rwm_column, replay):
    """States are stored in the MCMC space, as DREAM stores them and as their readers expect: solving the
    direct model with them as MCMC values gives back the recorded energy."""
    unique_states = {id(state): state for state in rwm_column._states}.values()  # rejections repeat a state
    inconsistent = [state.energy for state in unique_states
                    if replay(state)[0] != pytest.approx(state.energy, rel=ENERGY_RTOL)]
    assert not inconsistent, f"{len(inconsistent)} of {len(unique_states)} states, energies {inconsistent[:5]}"


def test_rwm_parameter_getters_return_physical_values_in_prior_range(rwm_column):
    """get_best_param and get_all_params convert the states to physical values inside the prior range."""
    lower, upper = rwm_column.all_layers[0].Prior_IntrinK.user_range
    best = rwm_column.get_best_param()[0].IntrinK
    every = [params[0].IntrinK for params in rwm_column.get_all_params()]
    assert lower <= best <= upper
    assert all(lower <= value <= upper for value in every)


def test_rwm_acceptance_rate_is_accepted_over_proposed(rwm_column):
    """The rate is the fraction of iterations where the state changed."""
    states = rwm_column._states
    changes = [not np.array_equal(state_after(states, i - 1).layers, state_after(states, i).layers)
               for i in range(N_ITER)]
    assert rwm_column.get_all_acceptance_ratio() == pytest.approx(np.mean(changes))


def test_rwm_quantiles_use_only_accepted_states(rwm_column, replay):
    """The quantiles are those of the fields of the accepted states.
    With these settings every iteration is kept for the quantiles."""
    fields = [replay(state_after(rwm_column._states, i))[1:] for i in range(N_ITER)]
    temperatures = np.array([t for t, _ in fields], np.float32)
    flows = np.array([f for _, f in fields], np.float32)

    for quantile in rwm_column.get_quantiles():
        np.testing.assert_allclose(rwm_column.get_temperatures_quantile(quantile),
                                   np.quantile(temperatures, quantile, axis=0), rtol=QUANTILE_RTOL)
        np.testing.assert_allclose(rwm_column.get_flows_quantile(quantile),
                                   np.quantile(flows, quantile, axis=0), rtol=QUANTILE_RTOL)
