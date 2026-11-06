"""The parameters an MCMC estimates, block by block, each with its Prior. For a pyheatmy column a block is a layer and
its parameters are those of pyheatmy's Param; the state space does not need to know it."""

from typing import Mapping

import numpy as np

from bayesian_inversion.prior import Prior


class StateSpace:
    """Named blocks of named parameters, each with its Prior: the samplers move in the MCMC space of the priors, the
    model receives physical values. blocks maps a block name to a mapping of parameter names to priors; every block
    has the same parameters, in the same order. A parameter can be fixed by its prior (Prior(value, 0))."""

    def __init__(self, blocks: Mapping[str, Mapping[str, Prior]]):
        if not blocks:
            raise ValueError("A state space needs at least one block of parameters.")
        self.block_names = list(blocks)
        self.param_names = list(next(iter(blocks.values())))
        for name, params in blocks.items():
            if list(params) != self.param_names:
                raise ValueError(
                    f"Block '{name}' has the parameters {list(params)}, the first block {self.param_names}: "
                    "every block needs the same parameters, in the same order."
                )
        self.priors = [[blocks[block][param] for param in self.param_names] for block in self.block_names]

    @property
    def shape(self):
        """(number of blocks, number of parameters per block): the shape of a state."""
        return len(self.block_names), len(self.param_names)

    def to_physical(self, x):
        """Physical values of a state x (one row per block, in the MCMC space), one mapping of parameter names to
        values per block: the argument of the model."""
        return [
            {name: prior.mcmc_to_physical(value) for name, prior, value in zip(self.param_names, priors, values)}
            for priors, values in zip(self.priors, x)
        ]

    def physical_array(self, x):
        """Physical values of a state x, as an array with the shape of the state."""
        return np.array(
            [[prior.mcmc_to_physical(value) for prior, value in zip(priors, values)] for priors, values in zip(self.priors, x)],
            np.float64,
        )

    def __repr__(self) -> str:
        return "\n".join(
            f"{block}: " + ", ".join(f"{name} {prior}" for name, prior in zip(self.param_names, priors))
            for block, priors in zip(self.block_names, self.priors)
        )
