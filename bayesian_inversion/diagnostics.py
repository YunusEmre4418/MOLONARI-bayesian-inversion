from numpy import full, inf, mean, nan, sqrt, var


def gelman_rubin(chains, is_param_fixed):
    """R_hat of Gelman and Rubin (1992) for each parameter of each layer, as computed by DREAM (Vrugt, 2016,
    Eqs. 32-35): from the last half of the samples of each chain, without the degrees of freedom factor.
    chains is a (sample, chain, layer, parameter) array, is_param_fixed a (layer, parameter) array of booleans.
    A fixed parameter has no R_hat (nan). A free parameter has an infinite R_hat when its chains do not move or
    when the last half has a single sample."""
    nb_sample, nb_chain = chains.shape[:2]
    n = nb_sample // 2
    last_half = chains[nb_sample - n :]
    r_hat = full(chains.shape[2:], inf)
    if n > 1:
        within = mean(var(last_half, axis=0, ddof=1), axis=0)  # W
        between = var(mean(last_half, axis=0), axis=0, ddof=1)  # B / n
        moving = within > 0
        sigma2_plus = (n - 1) / n * within + between
        r_hat[moving] = sqrt(
            (nb_chain + 1) / nb_chain * sigma2_plus[moving] / within[moving]
            - (n - 1) / (nb_chain * n)
        )
    r_hat[is_param_fixed] = nan
    return r_hat
