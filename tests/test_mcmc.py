"""Tests of run_mcmc and MCMCResult on a toy model, without pyheatmy: a straight line a + b t on one "cell", with
outputs named differently from pyheatmy's, so that nothing in run_mcmc depends on the model's package."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

from bayesian_inversion import Prior, StateSpace, compute_energy, run_mcmc

N_TIMES = 50
TIMES = np.linspace(0.0, 1.0, N_TIMES)
TRUE = {"a": 1.0, "b": 2.0}
NOISE = 0.1
SIGMA2 = NOISE**2
DATA_SEED = 20261106
SEED = 7
OTHER_SEED = 8
PRIOR_RANGE = (-5.0, 5.0)        # linear scale
STEP = 0.1
FIXED_C = 0.5                    # a fixed parameter, which the model ignores

N_CHAIN = 7                      # at least 2 * delta + 1 for the default delta = 3
N_ITER = 30
N_BURN_IN_MAX = 5
ONE_CHAIN = 1
QUANTILES = (0.05, 0.5, 0.95)
SPACE_STEP = 1
TIME_STEP = 2                    # one time out of two in the stored outputs
ENERGY_RTOL = 1e-12


def line(params):
    (block,) = params
    values = block["a"] + block["b"] * TIMES
    return {"predicted": values[np.newaxis, :], "field": values[np.newaxis, :]}


def observations():
    rng = np.random.default_rng(DATA_SEED)
    return line([TRUE])["predicted"] + rng.normal(0.0, NOISE, (1, N_TIMES))


def space():
    return StateSpace({"line": {"a": Prior(PRIOR_RANGE, STEP), "b": Prior(PRIOR_RANGE, STEP), "c": Prior(FIXED_C, 0)}})


def run(nb_chain, seed=SEED, **arguments):
    return run_mcmc(line, observations(), space(), predicted="predicted", stored=("field",), quantile=QUANTILES,
                    sigma2=SIGMA2, nb_iter=N_ITER, nb_chain=nb_chain, nitmaxburning=N_BURN_IN_MAX, seed=seed,
                    **arguments)


def uniform_density(_sigma2):
    return 1.0


@pytest.fixture(scope="module", params=[pytest.param(N_CHAIN, id="DREAM"), pytest.param(ONE_CHAIN, id="RWM")])
def result(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # short DREAM burn-in
        return run(request.param)


def test_states_per_chain_and_iteration(result):
    assert len(result.states) == result.nb_chain * (N_ITER + 1)
    assert result.chains().shape == (N_ITER + 1, result.nb_chain, *space().shape)


def test_every_energy_is_the_energy_of_its_parameters(result):
    data = observations()
    for state in {id(state): state for state in result.states}.values():
        output = line(space().to_physical(state.params))
        expected = compute_energy(output["predicted"], data, state.sigma2, uniform_density)
        assert state.energy == pytest.approx(expected, rel=ENERGY_RTOL)


def test_fixed_parameter_and_prior_range(result):
    params = result.params()
    assert np.all(params[:, 0, 2] == FIXED_C)
    assert np.all((PRIOR_RANGE[0] <= params[:, 0, :2]) & (params[:, 0, :2] <= PRIOR_RANGE[1]))
    best = result.best()[0]
    assert result.best_state().energy == result.energies().min()
    assert set(best) == {"a", "b", "c"}


def test_quantiles_of_the_stored_outputs(result):
    assert set(result.quantiles) == {"field"}
    assert set(result.quantiles["field"]) == set(QUANTILES)
    assert result.quantiles["field"][0.5].shape == (1, N_TIMES)


def test_stored_outputs_are_subsampled():
    subsampled = run(ONE_CHAIN, n_sous_ech_space=SPACE_STEP, n_sous_ech_time=TIME_STEP)
    assert subsampled.quantiles["field"][0.5].shape == (1, int(np.ceil(N_TIMES / TIME_STEP)))


@pytest.mark.parametrize("nb_chain", [pytest.param(N_CHAIN, id="DREAM"), pytest.param(ONE_CHAIN, id="RWM")])
@pytest.mark.filterwarnings("ignore:The DREAM burn-in")
def test_seed_gives_the_same_run(nb_chain):
    first, second, other = run(nb_chain), run(nb_chain), run(nb_chain, seed=OTHER_SEED)
    assert np.array_equal(first.energies(), second.energies())
    assert np.array_equal(first.params(), second.params())
    assert not np.array_equal(first.energies(), other.energies())


def test_run_mcmc_does_not_load_pyheatmy():
    code = "import sys, bayesian_inversion; assert 'pyheatmy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
