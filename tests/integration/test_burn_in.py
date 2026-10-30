"""Integration tests of the end of the DREAM burn-in (InversionColumn.compute_mcmc). The burn-in stops when the Gelman-Rubin
R_hat of every free parameter is at most the threshold, or after nitmaxburning iterations; a parameter fixed by its
prior does not move, so it has nothing to converge and must not hold the burn-in back. When the burn-in reaches
nitmaxburning without converging, the following states may not sample the posterior, and the user is warned."""

import random
import warnings

import numpy as np
import pytest

from pyheatmy import Layer
from pyheatmy.config import NSECINDAY, ZERO_CELSIUS
from pyheatmy.synthetic_MOLONARI import synthetic_MOLONARI

from bayesian_inversion.inversion_column import InversionColumn

pytestmark = pytest.mark.slow  # every test of this file needs an MCMC run

SEED = 0
DEPTH_SENSORS = [0.1, 0.2, 0.3, 0.4]
T_MEAN = ZERO_CELSIUS + 12
SERIES = {
    "offset": 0.0,
    "depth_sensors": DEPTH_SENSORS,
    "param_time_dates": [(2026, 1, 1, 0, 0, 0), (2026, 1, 2, 0, 0, 0), 900],   # 1 day, 96 times
    "param_dH_signal": [[0.1]],
    "param_T_riv_signal": [[T_MEAN], [2.0, NSECINDAY, 0.0]],
    "param_T_aq_signal": [[T_MEAN]],
    "sigma_meas_P": None,
    "sigma_meas_T": None,
    "verbose": False,
}
LAYER = {"name": "sand", "zLow": DEPTH_SENSORS[-1], "IntrinK": 1e-12, "n": 0.1,
         "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 0.0}
FIXED_PRIORS = {f"Prior_{name}": (LAYER[name], 0.0) for name in ("IntrinK", "n", "lambda_s", "rhos_cs", "q_s")}
FREE_POROSITY = {"Prior_n": ((0.05, 0.25), 0.01)}  # one free parameter, the others fixed
N_CELLS = 10
N_ITER = 2
N_BURN_IN_MAX = 4
N_CHAIN_DREAM = 7                                   # at least 2 * delta + 1 for the default delta = 3
BURN_IN_WARNING = "burn-in"


def column_with(priors, tmp_path):
    series = synthetic_MOLONARI.from_dict(SERIES)
    column = InversionColumn.from_dict({"river_bed": 1.0, "depth_sensors": DEPTH_SENSORS, "offset": 0.0,
                               "dH_measures": series._molonariP_data, "T_measures": series._molonariT_data,
                               "nb_cells": N_CELLS, "rac": str(tmp_path)})
    layer = Layer.from_dict(LAYER)
    layer.set_priors_from_dict(priors)
    column.set_layers(layer)
    return column


def run_dream(column, nitmaxburning):
    random.seed(SEED)
    np.random.seed(SEED)
    column.compute_mcmc(verbose=False, nb_iter=N_ITER, nb_chain=N_CHAIN_DREAM, nitmaxburning=nitmaxburning)


def burn_in_warnings(record):
    return [w for w in record if BURN_IN_WARNING in str(w.message)]


def test_fixed_parameters_do_not_hold_the_burn_in_back(tmp_path):
    column = column_with(FIXED_PRIORS, tmp_path)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        run_dream(column, N_BURN_IN_MAX)
    assert column.nb_burn_in_iter == 0               # nothing to converge: stops at the first check
    assert not burn_in_warnings(record)


def test_burn_in_without_convergence_warns(tmp_path):
    column = column_with({**FIXED_PRIORS, **FREE_POROSITY}, tmp_path)
    # One burn-in iteration gives two samples per chain: the last half has one, so R_hat of n is infinite.
    with pytest.warns(UserWarning, match=BURN_IN_WARNING) as record:
        run_dream(column, nitmaxburning=1)
    assert column.nb_burn_in_iter == 1
    message = str(burn_in_warnings(record)[0].message)
    assert "nitmaxburning" in message and "'n'" in message
