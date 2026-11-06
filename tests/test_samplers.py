"""Tests of the samplers (DREAM, RandomWalk) on a toy target, without pyheatmy: a Gaussian with independent
parameters, whose mean and variance are known. The model returns its parameters; the energy is the negative
log-density of the Gaussian and does not depend on sigma2."""

import random
import subprocess
import sys

import numpy as np
import pytest

from bayesian_inversion import Prior
from bayesian_inversion.samplers import DREAM, RandomWalk

SEED = 20261106
MEAN = np.array([1.0, -2.0])
STD = np.array([1.0, 0.5])
PRIOR_RANGE = (-10.0, 10.0)     # linear scale, far from the target: the wrapping at the bounds does not matter
PRIOR_STEP = 1.0                # sigma of the priors: the random walk Metropolis step, of the order of the target
                                # standard deviations
SIGMA2 = 1.0                    # sigma2 is not estimated: its prior is a single value, as with typealgo="no sigma"
FIXED_VALUE = 3.0

# DREAM: compute_mcmc defaults, with at least 2 * DELTA + 1 chains
N_CHAIN = 10
DELTA = 3
N_CR = 3
C = 0.1
C_STAR = 1e-12
DREAM_BURN_IN = 200             # adapted steps left out
DREAM_ITER = 2000               # steps kept: N_CHAIN * DREAM_ITER states
RWM_DRAWS = 25                  # draws from the priors for the start of the chain (NBBURNING)
RWM_BURN_IN = 1000
RWM_ITER = 20000

# Monte Carlo error: batch means over consecutive iterations, which accounts for the autocorrelation of the chains
N_BATCH = 20
N_STANDARD_ERRORS = 4

SHORT_ITER = 50                 # reproducibility and fixed parameters


def gaussian_energy(output, sigma2):
    return 0.5 * np.sum(((output[0] - MEAN) / STD) ** 2)


def model(x):
    return x.copy()


def priors(fixed_middle=False):
    if fixed_middle:  # a fixed parameter between the two moving ones; the energy reads the moving ones
        return [[Prior(PRIOR_RANGE, PRIOR_STEP), Prior(FIXED_VALUE, PRIOR_STEP), Prior(PRIOR_RANGE, PRIOR_STEP)]]
    return [[Prior(PRIOR_RANGE, PRIOR_STEP), Prior(PRIOR_RANGE, PRIOR_STEP)]]


def energy_without_fixed(output, sigma2):
    return gaussian_energy(output[:, [0, 2]], sigma2)


def sigma2_prior():
    return Prior((SIGMA2, SIGMA2), 0)


def seed():
    random.seed(SEED)
    np.random.seed(SEED)


def run_dream(n_iter, burn_in, fixed_middle=False):
    """Samples of the moving chains: array (n_iter, N_CHAIN, n_param)."""
    energy = energy_without_fixed if fixed_middle else gaussian_energy
    sampler = DREAM(model, energy, priors(fixed_middle), sigma2_prior(), N_CHAIN, DELTA, N_CR, C, C_STAR)
    sampler.initialize()
    for _ in range(burn_in):
        sampler.step(adapt=True)
    samples = []
    for _ in range(n_iter):
        sampler.step()
        samples.append(sampler.x[:, 0].copy())
    return np.array(samples)


def run_rwm(n_iter, burn_in, fixed_middle=False):
    """Samples of the chain: array (n_iter, 1, n_param)."""
    energy = energy_without_fixed if fixed_middle else gaussian_energy
    sampler = RandomWalk(model, energy, priors(fixed_middle), sigma2_prior())
    for _ in range(RWM_DRAWS):
        sampler.draw()
    for _ in range(burn_in):
        sampler.step()
    samples = []
    for _ in range(n_iter):
        sampler.step()
        samples.append(sampler.x[:, 0].copy())
    return np.array(samples)


def batch_mean_and_error(values):
    """Mean over iterations and chains of values (n_iter, n_chain, n_param), and its standard error from the
    means of N_BATCH batches of consecutive iterations."""
    batches = np.array([b.mean(axis=(0, 1)) for b in np.array_split(values, N_BATCH)])
    return values.mean(axis=(0, 1)), batches.std(axis=0, ddof=1) / np.sqrt(N_BATCH)


@pytest.mark.parametrize("run, n_iter, burn_in", [
    pytest.param(run_dream, DREAM_ITER, DREAM_BURN_IN, id="DREAM"),
    pytest.param(run_rwm, RWM_ITER, RWM_BURN_IN, id="RWM"),
])
def test_samples_have_the_mean_and_variance_of_the_target(run, n_iter, burn_in):
    seed()
    samples = run(n_iter, burn_in)
    mean, mean_error = batch_mean_and_error(samples)
    assert np.all(np.abs(mean - MEAN) < N_STANDARD_ERRORS * mean_error), (mean, mean_error)
    variance, variance_error = batch_mean_and_error((samples - MEAN) ** 2)
    assert np.all(np.abs(variance - STD**2) < N_STANDARD_ERRORS * variance_error), (variance, variance_error)


@pytest.mark.parametrize("run", [pytest.param(run_dream, id="DREAM"), pytest.param(run_rwm, id="RWM")])
def test_fixed_parameter_does_not_move(run):
    seed()
    samples = run(SHORT_ITER, burn_in=SHORT_ITER, fixed_middle=True)
    assert np.all(samples[:, :, 1] == FIXED_VALUE)
    assert np.all(np.ptp(samples[:, :, [0, 2]], axis=0) > 0)


@pytest.mark.parametrize("run", [pytest.param(run_dream, id="DREAM"), pytest.param(run_rwm, id="RWM")])
def test_same_seed_gives_same_chains(run):
    seed()
    first = run(SHORT_ITER, SHORT_ITER)
    seed()
    second = run(SHORT_ITER, SHORT_ITER)
    assert np.array_equal(first, second)


def test_dream_needs_two_groups_of_delta_other_chains():
    with pytest.raises(ValueError, match=r"at least 2 \* delta \+ 1"):
        DREAM(model, gaussian_energy, priors(), sigma2_prior(), 2 * DELTA, DELTA, N_CR, C, C_STAR)
    DREAM(model, gaussian_energy, priors(), sigma2_prior(), 2 * DELTA + 1, DELTA, N_CR, C, C_STAR)


def test_samplers_do_not_load_pyheatmy():
    code = "import sys, bayesian_inversion.samplers; assert 'pyheatmy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
