from dataclasses import dataclass
from typing import Sequence

from pyheatmy.params import Param

@dataclass
class State:
    # one Param per layer, in the MCMC space (Layer.mcmc_params);
    # convert with Column.get_all_params or Layer.get_physical_params

    layers: Sequence[Param]
    energy: float
    ratio_accept: float
    sigma2_temp: float
