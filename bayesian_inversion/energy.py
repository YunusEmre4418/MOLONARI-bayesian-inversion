import numpy as np
from numpy import log


def compute_energy(temp_simul, temp_ref, sigma2, sigma2_distrib):
    """Negative log-posterior of the temperature measurements: Gaussian errors of variance sigma2, without the
    constant (n/2) log(2 pi), and the prior density of sigma2. Missing measurements (NaN in temp_ref) are left out,
    and n is the number of measured values; a NaN of the model is not."""
    measured = ~np.isnan(temp_ref)
    norm2 = np.linalg.norm(temp_ref[measured] - temp_simul[measured]) ** 2
    return (
        (np.count_nonzero(measured) * log(sigma2) / 2)
        + norm2 / (2 * sigma2)
        - log(sigma2_distrib(sigma2))
    )


def compute_log_acceptance(current_energy: float, prev_energy: float):
    return prev_energy - current_energy
