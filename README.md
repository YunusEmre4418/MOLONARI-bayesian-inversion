# bayesian_inversion

Bayesian inversion of the [pyheatmy](https://github.com/YunusEmre4418/MOLONARI-pyheatmy) direct model: estimation
of the parameters of a riverbed column from temperature measurements, part of the MOLONARI project.

- `Prior`: range of a parameter, its sampling scale (linear, log or symlog, chosen from the range) and the
  transforms between the physical and MCMC spaces; a parameter can be fixed.
- `perturbation_DREAM`: the DREAM proposal (Vrugt, 2016) for one chain, with fixed parameters left out.
- `compute_energy`, `compute_log_acceptance`: negative log-posterior of Gaussian measurement errors, with missing
  measurements left out, and the Metropolis acceptance ratio.
- `gelman_rubin`: R_hat of Gelman and Rubin (1992), as computed by DREAM, for the burn-in.
- `Sampler`, `DREAM` (`bayesian_inversion.samplers`): the DREAM sampler, with several chains, on a common base
  for the samplers. It sees the model only through two functions, the model output for given parameters and its
  energy, and is run one step at a time by the caller.
- `InversionColumn` (`bayesian_inversion.inversion_column`): a `pyheatmy.Column` with the MCMC (DREAM with
  several chains, run by the DREAM sampler, random walk Metropolis with one), the records of its states, and
  the getters and plots of the posterior: `compute_mcmc`, `get_best_layers`, `get_temperatures_quantile`, ...

Only `inversion_column` imports `pyheatmy`; `import bayesian_inversion` does not.

## Installation and tests

`pyheatmy` is not on PyPI: install it first, from its checkout.

```sh
pip install -e path/to/pyheatmy
pip install -e .
python -m pytest tests            # about 2 minutes
python -m pytest tests -m "not slow"
```
