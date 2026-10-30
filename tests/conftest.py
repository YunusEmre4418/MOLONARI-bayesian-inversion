"""Shared pytest configuration for bayesian_inversion. pytest loads this file automatically."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs an MCMC, takes about a minute (skip with -m 'not slow')"
    )
