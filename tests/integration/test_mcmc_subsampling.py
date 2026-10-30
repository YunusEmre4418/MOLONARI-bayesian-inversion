"""Integration test of the subsampling of the MCMC states for the quantiles (InversionColumn.compute_mcmc): one sample is
stored every n_sous_ech_iter iterations, n_sous_ech_iter = nb_chain * nb_iter // NSAMPLEMIN, for any number of
iterations. With the layer parameters fixed, every state is the direct model, so every sample, and the minimum
and maximum of the samples, equal it."""

import random

import numpy as np
import pytest

from pyheatmy import Layer
from pyheatmy.config import NSECINDAY, ZERO_CELSIUS
from pyheatmy.synthetic_MOLONARI import synthetic_MOLONARI

from bayesian_inversion.config import NSAMPLEMIN
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
N_CELLS = 10
N_BURN_IN_MAX = 2
N_CHAIN_DREAM = 7                                 # at least 2 * delta + 1 for the default delta = 3
ONE_CHAIN = 1                                     # random walk Metropolis
QUANTILE_RTOL = 1e-6                              # float32 storage of the samples
SUBSAMPLING_STEP = 2                              # one stored sample every two iterations
EXTREME_QUANTILES = (0.0, 1.0)                    # minimum and maximum: any sample that is not a state shows


def fixed_column(tmp_path):
    series = synthetic_MOLONARI.from_dict(SERIES)
    column = InversionColumn.from_dict({"river_bed": 1.0, "depth_sensors": DEPTH_SENSORS, "offset": 0.0,
                               "dH_measures": series._molonariP_data, "T_measures": series._molonariT_data,
                               "nb_cells": N_CELLS, "rac": str(tmp_path)})
    layer = Layer.from_dict(LAYER)
    layer.set_priors_from_dict(FIXED_PRIORS)
    column.set_layers(layer)
    return column


def iterations(nb_chain, divisible):
    """Number of iterations giving a subsampling step of SUBSAMPLING_STEP, a multiple of the step or not."""
    nb_iter = SUBSAMPLING_STEP * NSAMPLEMIN // nb_chain + 1
    nb_iter += (-nb_iter) % SUBSAMPLING_STEP if divisible else (1 - nb_iter % SUBSAMPLING_STEP)
    assert nb_chain * nb_iter // NSAMPLEMIN == SUBSAMPLING_STEP
    return nb_iter


@pytest.mark.parametrize("nb_chain", [pytest.param(N_CHAIN_DREAM, id="DREAM"), pytest.param(ONE_CHAIN, id="RWM")])
@pytest.mark.parametrize("divisible", [pytest.param(True, id="multiple_of_step"),
                                       pytest.param(False, id="not_multiple_of_step")])
def test_quantiles_of_fixed_parameters_are_the_direct_model(nb_chain, divisible, tmp_path):
    column = fixed_column(tmp_path)
    column.compute_solve_transi(verbose=False)
    direct = np.asarray(column.get_temperatures_solve())

    random.seed(SEED)
    np.random.seed(SEED)
    column.compute_mcmc(quantile=EXTREME_QUANTILES, verbose=False, nb_iter=iterations(nb_chain, divisible),
                        nb_chain=nb_chain, nitmaxburning=N_BURN_IN_MAX)
    for quantile in column._quantiles_temperatures.values():
        assert quantile == pytest.approx(direct, rel=QUANTILE_RTOL)
