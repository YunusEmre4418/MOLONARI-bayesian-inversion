# bayesian_inversion

MCMC estimation of the parameters of a model from its measurements, part of the MOLONARI project. It was written
for the direct model of [pyheatmy](https://github.com/YunusEmre4418/MOLONARI-pyheatmy) (the parameters of the
layers of a riverbed column, from its temperatures), but it does not import `pyheatmy`: the model is any function.

```python
from bayesian_inversion import Prior, StateSpace, run_mcmc

space = StateSpace({"river bed": {"IntrinK": Prior((1e-15, 1e-10), 1e-12), "n": Prior(0.1, 0),
                                  "lambda_s": Prior((1.0, 5.0), 0.1), "rhos_cs": Prior(4e6, 0), "q_s": Prior(0.0, 0)}})
result = run_mcmc(column.simulate, column.get_measured_temperatures(), space, nb_chain=10, nb_iter=300, seed=1)
result.chains()                        # physical parameters, iteration by iteration and chain by chain
result.best()                          # parameters of the state of lowest energy, an argument of the model
result.quantiles["temperatures"][0.5]  # median of the stored outputs
```

- `StateSpace`: the estimated parameters, block by block (a layer of a column), each with its `Prior`.
- `run_mcmc(model, observations, state_space, ...)`: DREAM with several chains, random walk Metropolis with one.
  `model` takes one mapping of parameter names to physical values per block and returns a mapping of outputs:
  `output["sensors"]` is compared with `observations`, and the outputs `"temperatures"` and `"flows"` are stored for
  the quantiles (the names can be changed). `seed` seeds the random generators before the run.
- `MCMCResult`: the recorded states, their energies and sigma2, the acceptance rates, the quantiles, `best()`.
- `Prior`: range of a parameter, its sampling scale (linear, log or symlog, chosen from the range) and the
  transforms between the physical and MCMC spaces; a parameter can be fixed.
- `Sampler`, `DREAM`, `RandomWalk` (`bayesian_inversion.samplers`): the samplers `run_mcmc` uses. They see the
  model only through two functions, the model output for given parameters and its energy, and are run one step at
  a time by the caller.
- `perturbation_DREAM`: the DREAM proposal (Vrugt, 2016) for one chain, with fixed parameters left out.
- `compute_energy`, `compute_log_acceptance`: negative log-posterior of Gaussian measurement errors, with missing
  measurements left out, and the Metropolis acceptance ratio.
- `gelman_rubin`: R_hat of Gelman and Rubin (1992), as computed by DREAM, for the burn-in.

Until version 0.3.3, `bayesian_inversion.inversion_column.InversionColumn` was a `pyheatmy.Column` with the MCMC;
its `compute_mcmc(...)` is now `run_mcmc(column.simulate, column.get_measured_temperatures(), state_space, ...)`,
and the priors that were set on the layers are the state space.

## Installation and tests

```sh
pip install -e .
python -m pytest tests
```

The tests that use `bayesian_inversion` with `pyheatmy` are in the MOLONARI-computo tree
(`integration_tests/`, `pixi run test-all`).
