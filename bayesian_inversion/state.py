from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    # one row of parameters per block, in the MCMC space of the priors;
    # convert with StateSpace.to_physical or StateSpace.physical_array

    params: np.ndarray
    energy: float
    ratio_accept: float
    sigma2: float  # physical
