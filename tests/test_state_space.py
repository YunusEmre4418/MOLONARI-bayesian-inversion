"""Unit tests of StateSpace: the blocks and names of the parameters, and the physical values of a state."""

import numpy as np
import pytest

from bayesian_inversion import Prior, StateSpace

LOG_RANGE = (1e-15, 1e-10)       # log scale: the MCMC value is log10 of the physical one
LINEAR_RANGE = (1.0, 5.0)
STEP = 0.1
FIXED = 4e6
MCMC_STATE = np.array([[-12.0, 2.5, FIXED], [-13.0, 3.5, FIXED]])
PHYSICAL_STATE = np.array([[1e-12, 2.5, FIXED], [1e-13, 3.5, FIXED]])
RTOL = 1e-12


def two_blocks():
    block = {"IntrinK": Prior(LOG_RANGE, STEP), "lambda_s": Prior(LINEAR_RANGE, STEP), "rhos_cs": Prior(FIXED, 0)}
    return StateSpace({"top": block, "bottom": dict(block)})


def test_blocks_and_names_keep_their_order():
    space = two_blocks()
    assert space.block_names == ["top", "bottom"]
    assert space.param_names == ["IntrinK", "lambda_s", "rhos_cs"]
    assert space.shape == MCMC_STATE.shape


def test_physical_values_of_a_state():
    space = two_blocks()
    physical = space.to_physical(MCMC_STATE)
    assert [list(block) for block in physical] == [space.param_names] * len(space.block_names)
    np.testing.assert_allclose([list(block.values()) for block in physical], PHYSICAL_STATE, rtol=RTOL)
    np.testing.assert_allclose(space.physical_array(MCMC_STATE), PHYSICAL_STATE, rtol=RTOL)


def test_blocks_need_the_same_parameters_in_the_same_order():
    with pytest.raises(ValueError, match="same parameters"):
        StateSpace({"top": {"n": Prior(LINEAR_RANGE, STEP)}, "bottom": {"lambda_s": Prior(LINEAR_RANGE, STEP)}})


def test_a_state_space_needs_a_block():
    with pytest.raises(ValueError):
        StateSpace({})
