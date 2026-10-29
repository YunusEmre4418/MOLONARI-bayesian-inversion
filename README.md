# bayesian_inversion

Bayesian inversion tools used by [pyheatmy](https://github.com/YunusEmre4418/MOLONARI-pyheatmy) to estimate the
parameters of a riverbed column from temperature measurements, part of the MOLONARI project.

- `Prior`: range of a parameter, its sampling scale (linear, log or symlog, chosen from the range) and the
  transforms between the physical and MCMC spaces; a parameter can be fixed.
- `perturbation_DREAM`: the DREAM proposal (Vrugt, 2016) for one chain, with fixed parameters left out.
- `compute_energy`, `compute_log_acceptance`: negative log-posterior of Gaussian measurement errors, with missing
  measurements left out, and the Metropolis acceptance ratio.
- `gelman_rubin`: R_hat of Gelman and Rubin (1992), as computed by DREAM, for the burn-in.

The DREAM and random walk loops themselves are still in `pyheatmy.Column.compute_mcmc`.

`bayesian_inversion` does not import `pyheatmy`.

## Installation and tests

```sh
pip install -e .
python -m pytest tests
```
