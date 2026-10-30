"""Shared fixtures of the integration tests: the Point034 data set and the homogeneous column of the demo
notebook, and the replay of a recorded MCMC state through the direct model.
pytest loads this file automatically for every test of this folder."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyheatmy import Layer
from pyheatmy.config import ZERO_CELSIUS
from pyheatmy.params import Param

from bayesian_inversion import compute_energy
from bayesian_inversion.inversion_column import InversionColumn

TEST_DATA = Path(__file__).resolve().parents[1] / "data"  # copies of data/ and configuration/ of MOLONARI-computo
DATA_DIR = TEST_DATA / "Point034_processed"
PRESSURE_SENSOR_FILE = TEST_DATA / "P508.csv"
PRESSURE_SENSOR = "P508"
DATE_FORMAT = "%y/%m/%d %H:%M"
TEMPERATURE_COLUMNS = ["temperature_10", "temperature_20", "temperature_30", "temperature_40"]

SEED = 0  # seed of the reference runs of the refactoring

# Demo notebook: column, homogeneous layer and priors
DEMO_COLUMN = {
    "river_bed": 1.0,
    "depth_sensors": [0.1, 0.2, 0.3, 0.4],
    "offset": 0.0,
    "sigma_meas_P": None,
    "sigma_meas_T": None,
    "inter_mode": "lagrange",
    "nb_cells": 40,
}
DEMO_LAYER = {"name": "Layer_homogenous", "zLow": 0.4, "IntrinK": 1e-12, "n": 0.1,
              "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 0.0}
DEMO_PRIORS = {
    "Prior_IntrinK": ((1e-18, 1e-15), 5e-17),
    "Prior_n": ((0.01, 0.25), 0.0125),
    "Prior_lambda_s": ((1, 10), 0.5),
    "Prior_rhos_cs": ((1e6, 1e7), 9e5),
    "Prior_q_s": ((-1e-6, 1e-6), 1e-11),
}


def uniform_density(_sigma2):
    """Flat prior on sigma2, as compute_mcmc uses in the "no sigma" mode."""
    return 1.0


@pytest.fixture(scope="session")
def point034_measures():
    """River head and temperature, and sensor temperatures, processed as in the demo notebook."""
    river = pd.read_csv(DATA_DIR / "processed_pressures.csv",
                        names=["dates", "tension", "temperature_river"], skiprows=1)
    sensors = pd.read_csv(DATA_DIR / "processed_temperatures.csv",
                          names=["dates", *TEMPERATURE_COLUMNS], skiprows=1)
    calibration = pd.read_csv(PRESSURE_SENSOR_FILE, index_col=0)[PRESSURE_SENSOR]
    intercept = float(calibration["Intercept"])
    du_dh = float(calibration["dU/dH"])
    du_dt = float(calibration["dU/dT"])

    river["dates"] = pd.to_datetime(river["dates"], format=DATE_FORMAT)
    sensors["dates"] = pd.to_datetime(sensors["dates"], format=DATE_FORMAT)
    head = (river["tension"] - intercept - river["temperature_river"] * du_dt) / du_dh
    river_temperature = river["temperature_river"] + ZERO_CELSIUS
    sensor_temperatures = sensors[TEMPERATURE_COLUMNS].to_numpy() + ZERO_CELSIUS

    dH_measures = list(zip(river["dates"], zip(head, river_temperature)))
    T_measures = list(zip(sensors["dates"], sensor_temperatures))
    return dH_measures, T_measures


@pytest.fixture(scope="session")
def new_demo_column(point034_measures, tmp_path_factory):
    """Factory: each call builds a fresh demo column with its layer and priors.
    Output files of the column go to a temporary folder, not to the home directory."""
    dH_measures, T_measures = point034_measures

    def build():
        column = InversionColumn.from_dict({
            **DEMO_COLUMN,
            "dH_measures": dH_measures,
            "T_measures": T_measures,
            "rac": str(tmp_path_factory.mktemp("column_output")),
        })
        layer = Layer.from_dict(DEMO_LAYER)
        layer.set_priors_from_dict(DEMO_PRIORS)
        column.set_layers(layer)
        return column

    return build


@pytest.fixture(scope="session")
def run_demo_mcmc(new_demo_column):
    """Factory: each call runs compute_mcmc with the given arguments on a fresh demo column, from the same seed,
    and returns the column. Prior uses the random module and DREAM uses numpy, so both are seeded."""
    def run(**mcmc_arguments):
        random.seed(SEED)
        np.random.seed(SEED)
        column = new_demo_column()
        column.compute_mcmc(verbose=False, **mcmc_arguments)
        return column

    return run


@pytest.fixture(scope="session")
def replay(new_demo_column):
    """Function: solves the direct model on a separate column with the MCMC parameters of a state,
    and returns its energy, temperatures and flows."""
    column = new_demo_column()

    def solve(state):
        for layer, values in zip(column.all_layers, state.layers):
            layer.mcmc_params = Param(*values)
        column.compute_solve_transi(verbose=False)
        simulated = column.temperatures_solve[column.get_id_sensors(), :]
        energy = compute_energy(simulated, column._T_measures.T, state.sigma2_temp, uniform_density)
        return energy, np.asarray(column.get_temperatures_solve()), np.asarray(column.get_flows_solve())

    return solve
