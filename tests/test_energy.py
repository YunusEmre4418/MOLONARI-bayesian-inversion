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


# Missing measurements (NaN), moved from pyheatmy's test_missing_measurements.py: the likelihood covers the measured
# values only, so a NaN gives the same energy as the same data without it; a NaN of the model is not left out.
MIDDLE_SENSOR = 1
MISSING_TIMES = [10, 11, 50]
FIRST = 0
RTOL = 1e-12


def test_energy_leaves_out_missing_measurements():
    rng = np.random.default_rng(0)
    measured = T_MEASURED_K + rng.normal(size=(N_SENSORS, N_TIMES))
    simulated = measured + SHIFT_K
    with_gaps = measured.copy()
    with_gaps[MIDDLE_SENSOR, MISSING_TIMES] = np.nan
    kept = ~np.isnan(with_gaps)
    expected = compute_energy(simulated[kept], measured[kept], sigma2=SIGMA2, sigma2_distrib=uniform_density)
    energy = compute_energy(simulated, with_gaps, sigma2=SIGMA2, sigma2_distrib=uniform_density)
    assert energy == pytest.approx(expected, rel=RTOL)


def test_energy_is_not_finite_when_the_model_is_not():
    measured = np.full((N_SENSORS, len(MISSING_TIMES)), T_MEASURED_K)
    simulated = measured + SHIFT_K
    simulated[MIDDLE_SENSOR, FIRST] = np.nan       # a failed direct model must not look like a missing measurement
    assert np.isnan(compute_energy(simulated, measured, sigma2=SIGMA2, sigma2_distrib=uniform_density))
