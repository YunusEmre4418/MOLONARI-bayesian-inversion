"""Unit tests for the MCMC energy (negative log-posterior of the temperature measurements)."""
import numpy as np
import pytest

from bayesian_inversion import compute_energy

N_SENSORS = 3
N_TIMES = 100
T_MEASURED_K = 291.0
SHIFT_K = 0.5        # simulated = measured + SHIFT_K everywhere; not 1, since 1**2 == 1 hides mistakes
SIGMA2_ONE = 1.0     # log(1) = 0: the log(sigma2) term vanishes
SIGMA2 = 2.0


def uniform_density(_sigma2):
    """Flat prior on sigma2: its -log term is zero."""
    return 1.0


def shifted_measurements():
    measured = np.full((N_SENSORS, N_TIMES), T_MEASURED_K)
    return measured + SHIFT_K, measured


def test_energy_with_unit_sigma2_is_half_sum_of_squares():
    simulated, measured = shifted_measurements()
    E = compute_energy(simulated, measured, sigma2=SIGMA2_ONE, sigma2_distrib=uniform_density)
    # sigma2 = 1: E = sum(r**2) / 2 = n * SHIFT_K**2 / 2
    assert E == pytest.approx(measured.size * SHIFT_K**2 / 2)


def test_energy_matches_gaussian_negative_log_likelihood():
    simulated, measured = shifted_measurements()
    n = measured.size
    E = compute_energy(simulated, measured, sigma2=SIGMA2, sigma2_distrib=uniform_density)
    # -log L = (n/2) log(sigma2) + sum(r**2) / (2 sigma2), without the constant (n/2) log(2 pi)
    expected = (n / 2) * np.log(SIGMA2) + n * SHIFT_K**2 / (2 * SIGMA2)
    assert E == pytest.approx(expected)
