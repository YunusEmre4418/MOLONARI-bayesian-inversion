"""Unit tests for the DREAM proposal (perturbation_DREAM): the jump scale gamma,
the fixed parameters and the prior ranges."""

import numpy as np
import pytest

from bayesian_inversion import perturbation_DREAM
from bayesian_inversion.config import DREAM_GAMMA_MODE_JUMP, DREAM_PROB_MODE_JUMP, GAMMA_FACTOR

SEED = 20261023
N_PARAM = 5
N_CR = 3                        # compute_mcmc default
CR_VALUES = np.arange(1, N_CR + 1) / N_CR
CR_PROBABILITIES = np.full(N_CR, 1 / N_CR)

# Gamma: with c = c* = 0 the proposal is dX = gamma * (X_a - X_b), so gamma can be read back.
# Three chains and delta = 1: the pair drawn for chain 0 is always {1, 2}.
GAMMA_N_CHAIN = 3
GAMMA_DELTA = 1
GAMMA_CHAIN = 0
CHAIN_1 = np.arange(1, N_PARAM + 1, dtype=np.float64)  # 1 … 5
CHAIN_2 = -CHAIN_1                                       # |X1 - X2| = 2, 4, 6, 8, 10
WIDE_RANGE = (-1e6, 1e6)        # far from every proposal: no wrapping
NO_NOISE = 0.0
N_GAMMA_DRAWS = 10000
N_STANDARD_ERRORS = 4           # tolerance of the frequency test
CHAINS_PER_DIFFERENCE = 2       # gamma = GAMMA_FACTOR / sqrt(2 * delta * d*) (ter Braak 2006)
RELATIVE_TOLERANCE = 1e-12      # proposals are float64

# Realistic proposals: compute_mcmc defaults, ranges in the MCMC space of the demo priors
N_CHAIN = 7                     # at least 2 * DELTA + 1
DELTA = 3
C = 0.1
C_STAR = 1e-12
MCMC_RANGES = np.array([
    (-18.0, -15.0),             # log10 IntrinK
    (0.01, 0.25),               # n
    (1.0, 10.0),                # lambda_s
    (1e6, 1e7),                 # rhos_cs
    (-3.0, 3.0),                # q_s, symlog
])
N_PROPOSALS = 2000
FIXED_IN_FIRST_LAYER = np.array([False, True, False, True, False])


def propose(X, ranges, is_param_fixed, chain, delta, c, c_star):
    """One DREAM proposal for one chain."""
    nb_chain, nb_layer, nb_param = X.shape
    cr_probabilities = np.tile(CR_PROBABILITIES, (nb_layer, 1))
    id_layer = np.zeros(nb_layer, np.int32)
    X_proposal, dX, _ = perturbation_DREAM(
        nb_chain, nb_layer, nb_param, X, id_layer, chain,
        delta, N_CR, c, c_star, CR_VALUES, cr_probabilities, ranges, is_param_fixed,
    )
    return X_proposal, dX


def random_chains(n_chain, ranges):
    """Chains drawn uniformly inside the ranges, one layer per row of ranges."""
    lower, upper = ranges[..., 0], ranges[..., 1]
    return np.random.uniform(lower, upper, (n_chain, *lower.shape)).astype(np.float64)


@pytest.fixture(scope="module")
def gamma_draws():
    """Gamma and number of moved parameters d* of N_GAMMA_DRAWS proposals.
    scope="module": computed once, shared by the tests of this file."""
    np.random.seed(SEED)
    X = np.zeros((GAMMA_N_CHAIN, 1, N_PARAM), np.float64)
    X[1, 0], X[2, 0] = CHAIN_1, CHAIN_2
    ranges = np.tile(WIDE_RANGE, (1, N_PARAM, 1))
    is_param_fixed = np.zeros((1, N_PARAM), bool)
    difference = np.abs(CHAIN_1 - CHAIN_2)

    gammas, d_stars = [], []
    for _ in range(N_GAMMA_DRAWS):
        _, dX = propose(X, ranges, is_param_fixed, GAMMA_CHAIN, GAMMA_DELTA, NO_NOISE, NO_NOISE)
        moved = dX[0] != 0
        gamma = np.abs(dX[0][moved]) / difference[moved]
        assert gamma == pytest.approx(gamma[0], rel=RELATIVE_TOLERANCE)  # one gamma per proposal
        gammas.append(gamma[0])
        d_stars.append(moved.sum())
    return np.array(gammas), np.array(d_stars)


def test_proposal_moves_at_least_one_parameter(gamma_draws):
    """The crossover always selects at least one parameter, so no proposal is a copy of the chain."""
    _, d_stars = gamma_draws
    assert d_stars.min() >= 1


def test_mode_jump_gamma_is_used_with_config_probability(gamma_draws):
    """gamma = DREAM_GAMMA_MODE_JUMP in a fraction DREAM_PROB_MODE_JUMP of the proposals,
    within N_STANDARD_ERRORS binomial standard errors."""
    gammas, _ = gamma_draws
    is_mode_jump = np.isclose(gammas, DREAM_GAMMA_MODE_JUMP, rtol=RELATIVE_TOLERANCE)
    standard_error = np.sqrt(DREAM_PROB_MODE_JUMP * (1 - DREAM_PROB_MODE_JUMP) / N_GAMMA_DRAWS)
    assert is_mode_jump.mean() == pytest.approx(DREAM_PROB_MODE_JUMP, abs=N_STANDARD_ERRORS * standard_error)


def test_other_gammas_follow_jump_scale_formula(gamma_draws):
    """Every gamma that is not the mode jump one is GAMMA_FACTOR / sqrt(2 * delta * d*)."""
    gammas, d_stars = gamma_draws
    is_mode_jump = np.isclose(gammas, DREAM_GAMMA_MODE_JUMP, rtol=RELATIVE_TOLERANCE)
    expected = GAMMA_FACTOR / np.sqrt(CHAINS_PER_DIFFERENCE * GAMMA_DELTA * d_stars)
    assert gammas[~is_mode_jump] == pytest.approx(expected[~is_mode_jump], rel=RELATIVE_TOLERANCE)


def test_fixed_parameters_are_never_moved():
    """Fixed parameters keep their value, in a partly fixed layer and in a fully fixed one."""
    np.random.seed(SEED)
    ranges = np.stack([MCMC_RANGES, MCMC_RANGES])
    is_param_fixed = np.stack([FIXED_IN_FIRST_LAYER, np.ones(N_PARAM, bool)])
    X = random_chains(N_CHAIN, ranges)
    X[:, is_param_fixed] = X[0, is_param_fixed]          # a fixed value is the same in every chain
    ranges[is_param_fixed] = X[0, is_param_fixed, None]  # and its range has zero width

    for step in range(N_PROPOSALS):
        chain = step % N_CHAIN
        X_proposal, dX = propose(X, ranges, is_param_fixed, chain, DELTA, C, C_STAR)
        assert np.all(dX[is_param_fixed] == 0)
        assert np.all(X_proposal[is_param_fixed] == X[chain][is_param_fixed])
        X[chain] = X_proposal


def test_proposals_stay_in_prior_ranges():
    """Every proposal is wrapped back into the prior range, in the MCMC space."""
    np.random.seed(SEED)
    ranges = MCMC_RANGES[None]
    is_param_fixed = np.zeros((1, N_PARAM), bool)
    X = random_chains(N_CHAIN, ranges)

    for step in range(N_PROPOSALS):
        chain = step % N_CHAIN
        X_proposal, _ = propose(X, ranges, is_param_fixed, chain, DELTA, C, C_STAR)
        assert np.all(X_proposal >= ranges[..., 0])
        assert np.all(X_proposal <= ranges[..., 1])
        X[chain] = X_proposal
