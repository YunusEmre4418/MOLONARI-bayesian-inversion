"""Unit tests of the Gelman-Rubin diagnostic of the DREAM burn-in (bayesian_inversion.gelman_rubin).

The expected values are computed by hand from Gelman and Rubin (1992), steps 1 to 4 and 6 without the degrees of
freedom factor, as used by DREAM (Vrugt, 2016, Eqs. 32-35): with m chains and the last n of the 2n samples of
each chain, B / n is the variance of the m chain means and W the mean of the m within-chain variances, both with
one degree of freedom less, sigma2_plus = (n - 1) / n * W + B / n, and
R_hat ** 2 = (m + 1) / m * sigma2_plus / W - (n - 1) / (m * n).
The chains array has the layout of the burn-in history: (sample, chain, layer, parameter)."""

import math
import warnings

import numpy as np
import pytest

from bayesian_inversion import gelman_rubin

# Two chains of four samples; the first half is discarded, so its values must not matter.
FIRST_HALF = [[9.0, -5.0], [9.0, 7.0]]           # rows: samples 1-2, columns: chains 1-2
SPREAD_LAST_HALF = [[1.0, 5.0], [3.0, 7.0]]       # chain means 2 and 6, within-chain variances 2 and 2
# W = 2, B / n = ((2 - 4) ** 2 + (6 - 4) ** 2) / (2 - 1) = 8, n = 2, m = 2
# sigma2_plus = 1 / 2 * 2 + 8 = 9, R_hat ** 2 = 3 / 2 * 9 / 2 - 1 / 4 = 13 / 2
R_HAT_SPREAD = math.sqrt(13 / 2)
MIXED_LAST_HALF = [[1.0, 3.0], [3.0, 1.0]]        # chain means 3 and 3, within-chain variances 2 and 2
# W = 2, B / n = 0, sigma2_plus = 1, R_hat ** 2 = 3 / 2 * 1 / 2 - 1 / 4 = 1 / 2
R_HAT_MIXED = math.sqrt(1 / 2)
SMALL_SCALE = 1e-12                               # e.g. a flux in m/s sampled on a linear scale
FIXED_VALUE = 0.5
INITIAL_POPULATION = 1                            # the burn-in history before its first iteration
TOO_FEW_SAMPLES = 3                               # last half of one sample: no within-chain variance
RTOL = 1e-12                                      # double precision, a few operations


def chains_of(*parameters, layers=1):
    """(sample, chain, layer, parameter) array; each parameter is a (sample, chain) list, the same in every layer."""
    values = np.stack([np.asarray(p, dtype=float) for p in parameters], axis=-1)
    return np.repeat(values[:, :, np.newaxis, :], layers, axis=2)


def free(nb_layer, nb_param):
    return np.zeros((nb_layer, nb_param), dtype=bool)


def test_r_hat_uses_the_last_half_of_the_samples():
    chains = chains_of(FIRST_HALF + SPREAD_LAST_HALF, FIRST_HALF + MIXED_LAST_HALF)
    r_hat = gelman_rubin(chains, free(1, 2))
    assert r_hat.shape == (1, 2)
    assert r_hat[0] == pytest.approx([R_HAT_SPREAD, R_HAT_MIXED], rel=RTOL)


def test_r_hat_does_not_depend_on_the_unit_of_the_parameter():
    spread = np.asarray(FIRST_HALF + SPREAD_LAST_HALF)
    chains = chains_of(spread, spread * SMALL_SCALE)
    r_hat = gelman_rubin(chains, free(1, 2))
    assert r_hat[0] == pytest.approx([R_HAT_SPREAD, R_HAT_SPREAD], rel=RTOL)


def test_r_hat_of_each_layer():
    spread = chains_of(FIRST_HALF + SPREAD_LAST_HALF)
    mixed = chains_of(FIRST_HALF + MIXED_LAST_HALF)
    chains = np.concatenate([spread, mixed], axis=2)  # layer 0 spread, layer 1 mixed
    r_hat = gelman_rubin(chains, free(2, 1))
    assert r_hat[:, 0] == pytest.approx([R_HAT_SPREAD, R_HAT_MIXED], rel=RTOL)


def test_fixed_parameter_has_no_r_hat_and_stuck_chains_have_an_infinite_one():
    constant = np.full_like(np.asarray(FIRST_HALF + SPREAD_LAST_HALF), FIXED_VALUE)
    chains = chains_of(FIRST_HALF + SPREAD_LAST_HALF, constant, constant)
    is_fixed = free(1, 3)
    is_fixed[0, 1] = True                         # parameter 1 fixed by its prior, parameter 2 free but stuck
    r_hat = gelman_rubin(chains, is_fixed)
    assert r_hat[0, 0] == pytest.approx(R_HAT_SPREAD, rel=RTOL)
    assert np.isnan(r_hat[0, 1])
    assert r_hat[0, 2] == math.inf


@pytest.mark.parametrize("nb_sample", [INITIAL_POPULATION, TOO_FEW_SAMPLES])
def test_too_few_samples_give_an_infinite_r_hat_without_numpy_warnings(nb_sample):
    chains = chains_of((FIRST_HALF + SPREAD_LAST_HALF)[:nb_sample])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no variance of fewer than two samples is computed
        r_hat = gelman_rubin(chains, free(1, 1))
    assert r_hat[0, 0] == math.inf
