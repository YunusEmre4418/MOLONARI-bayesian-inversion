"""Unit tests for Prior (bayesian_inversion.prior): the choice of the sampling scale from the physical range,
the transforms between the physical and MCMC spaces, and the random draws and steps."""

import random

import pytest

from bayesian_inversion import Prior, config
from bayesian_inversion import prior as prior_module

USER_SIGMA = 0.5  # does not affect the scale choice
# Physical values go down to 1e-17, below the default absolute tolerance of pytest.approx (1e-12),
# so comparisons of physical values use a relative tolerance only (abs=0).
RELATIVE_TOLERANCE = 1e-9

# Typical ranges
INTRINK_RANGE = (1e-18, 1e-15)      # permeability in m2: three orders of magnitude
N_RANGE = (0.01, 0.25)              # porosity: ratio 25, below the log threshold
Q_S_RANGE = (-1e-6, 1e-6)           # source term: crosses zero, small magnitude
WIDE_ZERO_CROSSING_RANGE = (-5, 5)  # crosses zero, large magnitude
FIXED_VALUE = 0.3

# Threshold cases
LOWER = 2.0  # not 1: a unit lower bound hides mistakes in the ratio
JUST_ABOVE = 1.01
JUST_BELOW = 0.99

MAGNITUDE_JUST_ABOVE_SYMLOG = config.SYMLOG_MAGNITUDE_THRESHOLD * JUST_ABOVE
MAGNITUDE_JUST_BELOW_SYMLOG = config.SYMLOG_MAGNITUDE_THRESHOLD * JUST_BELOW

RANGE_JUST_ABOVE_LOG_THRESHOLD = (LOWER, LOWER * config.LOG_SCALE_THRESHOLD * JUST_ABOVE)
RANGE_JUST_BELOW_LOG_THRESHOLD = (LOWER, LOWER * config.LOG_SCALE_THRESHOLD * JUST_BELOW)
RANGE_JUST_ABOVE_SYMLOG_THRESHOLD = (-MAGNITUDE_JUST_ABOVE_SYMLOG, MAGNITUDE_JUST_ABOVE_SYMLOG)
RANGE_JUST_BELOW_SYMLOG_THRESHOLD = (-MAGNITUDE_JUST_BELOW_SYMLOG, MAGNITUDE_JUST_BELOW_SYMLOG)

# Transforms
INTRINK_VALUE = 3e-17
N_VALUE = 0.12
Q_S_LOG_REGION_VALUE = 5e-7          # above the linear threshold of Q_S_RANGE (1e-6 / 100 = 1e-8)
Q_S_NEAR_THRESHOLD_VALUE = 3e-8      # between linthresh and 10 linthresh: MCMC value between 1 and 2
Q_S_LINEAR_REGION_VALUE = 3e-9       # below the linear threshold

LOG_VALUE = 1e-16
LOG_VALUE_EXPONENT = -16
INTRINK_RANGE_EXPONENTS = (-18, -15)

MCMC_VALUE_AT_LINTHRESH = 1.0  # symlog maps +-linthresh to +-1, where its two branches meet
TIGHT_JUST_ABOVE = 1 + 1e-9
TIGHT_JUST_BELOW = 1 - 1e-9
CONTINUITY_TOLERANCE = 1e-6     # a jump at linthresh would be of order 1

# Random draws and steps
SEED = 20261023
N_DRAWS = 1000
STEP_RANGE = (2.0, 12.0)        # linear: the MCMC space is the physical space, span 10
LARGE_STEP_SIGMA = 25.0         # larger than the span, so that most steps wrap
NEAR_UPPER_FRACTION = 0.9       # start 10 % below the upper bound
NEAR_LOWER_FRACTION = 0.1       # start 10 % above the lower bound
STEP_FRACTION = 0.3             # step of 30 % of the span
WRAPPED_UP_FRACTION = 0.2       # by hand: 0.9 + 0.3 = 1.2, one span too far, lands at 0.2
WRAPPED_DOWN_FRACTION = 0.8     # by hand: 0.1 - 0.3 = -0.2, one span too low, lands at 0.8
OUTSIDE_VALUE = 7.0             # any value different from the fixed one


def at_fraction(user_range, fraction):
    """Point of a linear range at the given fraction of its span, from the lower bound."""
    lower, upper = user_range
    return lower + fraction * (upper - lower)


@pytest.mark.parametrize("user_range, expected_scale", [
    pytest.param(INTRINK_RANGE, "log", id="permeability"),
    pytest.param(N_RANGE, "linear", id="porosity"),
    pytest.param(Q_S_RANGE, "symlog", id="source_term"),
    pytest.param(WIDE_ZERO_CROSSING_RANGE, "linear", id="wide_zero_crossing"),
    pytest.param(FIXED_VALUE, "fixed", id="fixed"),
])
def test_prior_chooses_scale_for_typical_ranges(user_range, expected_scale):
    """Typical physical ranges get the documented scale: log, linear, symlog or fixed."""
    prior = Prior(user_range, USER_SIGMA)
    assert prior.scale == expected_scale


@pytest.mark.parametrize("user_range, expected_scale", [
    pytest.param(RANGE_JUST_ABOVE_LOG_THRESHOLD, "log", id="just_above_log_threshold"),
    pytest.param(RANGE_JUST_BELOW_LOG_THRESHOLD, "linear", id="just_below_log_threshold"),
    pytest.param(RANGE_JUST_ABOVE_SYMLOG_THRESHOLD, "linear", id="just_above_symlog_threshold"),
    pytest.param(RANGE_JUST_BELOW_SYMLOG_THRESHOLD, "symlog", id="just_below_symlog_threshold"),
])
def test_prior_scale_switches_at_config_thresholds(user_range, expected_scale):
    """Each scale rule switches at its config threshold, checked on both sides."""
    prior = Prior(user_range, USER_SIGMA)
    assert prior.scale == expected_scale


@pytest.mark.parametrize("user_range, physical_value", [
    pytest.param(INTRINK_RANGE, INTRINK_VALUE, id="log"),
    pytest.param(N_RANGE, N_VALUE, id="linear"),
    pytest.param(Q_S_RANGE, Q_S_LOG_REGION_VALUE, id="symlog_log_region_positive"),
    pytest.param(Q_S_RANGE, -Q_S_LOG_REGION_VALUE, id="symlog_log_region_negative"),
    pytest.param(Q_S_RANGE, Q_S_NEAR_THRESHOLD_VALUE, id="symlog_log_region_near_threshold"),
    pytest.param(Q_S_RANGE, Q_S_LINEAR_REGION_VALUE, id="symlog_linear_region_positive"),
    pytest.param(Q_S_RANGE, -Q_S_LINEAR_REGION_VALUE, id="symlog_linear_region_negative"),
])
def test_mcmc_to_physical_inverts_physical_to_mcmc(user_range, physical_value):
    """Going to the MCMC space and back gives the physical value again, on every scale."""
    prior = Prior(user_range, USER_SIGMA)
    round_trip = prior.mcmc_to_physical(prior.physical_to_mcmc(physical_value))
    assert round_trip == pytest.approx(physical_value, rel=RELATIVE_TOLERANCE, abs=0)


def test_log_scale_works_with_decimal_exponents():
    """On the log scale, the MCMC value is the base 10 exponent, for values and for bounds."""
    prior = Prior(INTRINK_RANGE, USER_SIGMA)
    assert prior.physical_to_mcmc(LOG_VALUE) == pytest.approx(LOG_VALUE_EXPONENT)
    assert tuple(prior.mcmc_range) == pytest.approx(INTRINK_RANGE_EXPONENTS)


def test_symlog_linear_threshold_is_largest_bound_over_ratio():
    """The linear zone of symlog is set by the largest bound, as the code comment documents."""
    prior = Prior(Q_S_RANGE, USER_SIGMA)
    largest_bound = max(abs(bound) for bound in Q_S_RANGE)
    assert prior.linthresh == pytest.approx(largest_bound / config.SYMLOG_LINTHRESH_RATIO,
                                            rel=RELATIVE_TOLERANCE, abs=0)


def test_symlog_is_continuous_at_linear_threshold():
    """The linear and log branches of symlog meet at linthresh, in both directions of the transform."""
    prior = Prior(Q_S_RANGE, USER_SIGMA)
    below = prior.physical_to_mcmc(prior.linthresh * TIGHT_JUST_BELOW)
    above = prior.physical_to_mcmc(prior.linthresh * TIGHT_JUST_ABOVE)
    assert below == pytest.approx(MCMC_VALUE_AT_LINTHRESH, abs=CONTINUITY_TOLERANCE)
    assert above == pytest.approx(MCMC_VALUE_AT_LINTHRESH, abs=CONTINUITY_TOLERANCE)

    back_below = prior.mcmc_to_physical(MCMC_VALUE_AT_LINTHRESH * TIGHT_JUST_BELOW)
    back_above = prior.mcmc_to_physical(MCMC_VALUE_AT_LINTHRESH * TIGHT_JUST_ABOVE)
    assert back_below == pytest.approx(prior.linthresh, rel=CONTINUITY_TOLERANCE, abs=0)
    assert back_above == pytest.approx(prior.linthresh, rel=CONTINUITY_TOLERANCE, abs=0)


@pytest.mark.parametrize("user_range", [
    pytest.param(INTRINK_RANGE, id="log"),
    pytest.param(N_RANGE, id="linear"),
    pytest.param(Q_S_RANGE, id="symlog"),
])
def test_sample_stays_in_mcmc_range(user_range):
    """Initial draws always fall inside the prior range, expressed in the MCMC space."""
    random.seed(SEED)
    prior = Prior(user_range, USER_SIGMA)
    lower, upper = prior.mcmc_range
    draws = [prior.sample() for _ in range(N_DRAWS)]
    assert all(lower <= draw <= upper for draw in draws)


def test_perturb_stays_in_mcmc_range_with_large_steps():
    """Steps larger than the range are wrapped back into it, whatever the starting point."""
    random.seed(SEED)
    prior = Prior(STEP_RANGE, LARGE_STEP_SIGMA)
    lower, upper = prior.mcmc_range
    value = prior.sample()
    for _ in range(N_DRAWS):
        value = prior.perturb(value)
        assert lower <= value <= upper


@pytest.mark.parametrize("start_fraction, step_fraction, expected_fraction", [
    pytest.param(NEAR_UPPER_FRACTION, STEP_FRACTION, WRAPPED_UP_FRACTION, id="out_through_upper"),
    pytest.param(NEAR_LOWER_FRACTION, -STEP_FRACTION, WRAPPED_DOWN_FRACTION, id="out_through_lower"),
])
def test_perturb_wraps_around_range(monkeypatch, start_fraction, step_fraction, expected_fraction):
    """A step that leaves the range comes back in from the other side ("Pac-Man")."""
    span = STEP_RANGE[1] - STEP_RANGE[0]
    # Replace the random step by a known one; prior looks gauss up in its own namespace.
    monkeypatch.setattr(prior_module, "gauss", lambda _mu, _sigma: step_fraction * span)
    prior = Prior(STEP_RANGE, USER_SIGMA)
    new_value = prior.perturb(at_fraction(STEP_RANGE, start_fraction))
    assert new_value == pytest.approx(at_fraction(STEP_RANGE, expected_fraction))


def test_fixed_prior_never_moves():
    """A prior given a single number always samples and perturbs to that number."""
    prior = Prior(FIXED_VALUE, USER_SIGMA)
    assert prior.is_fixed
    assert prior.sample() == FIXED_VALUE
    assert prior.perturb(OUTSIDE_VALUE) == FIXED_VALUE


def test_zero_width_range_is_not_fixed_and_keeps_any_value():
    """Documents current behaviour: (x, x) is not treated as fixed, and perturb returns its input
    unchanged, even outside the range. core.py builds sigma2 priors this way."""
    prior = Prior((FIXED_VALUE, FIXED_VALUE), USER_SIGMA)
    assert not prior.is_fixed
    assert prior.scale == "linear"
    assert prior.sample() == FIXED_VALUE
    assert prior.perturb(OUTSIDE_VALUE) == OUTSIDE_VALUE
