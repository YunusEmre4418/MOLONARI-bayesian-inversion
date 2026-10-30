# Thresholds of the sampling scale chosen by Prior (linear, log or symlog)
LOG_SCALE_THRESHOLD= 1000**0.5 # Moyenne géométrique entre 1e1 et 1e2
SYMLOG_MAGNITUDE_THRESHOLD = 0.1
SYMLOG_LINTHRESH_RATIO= 1e2

GAMMA_FACTOR = 2.38  # DREAM jump scale: gamma = GAMMA_FACTOR / sqrt(2 * delta * d*)
DREAM_PROB_MODE_JUMP = 0.2   # DREAM: probability of using gamma = DREAM_GAMMA_MODE_JUMP, to jump between posterior modes
DREAM_GAMMA_MODE_JUMP = 1.0

# Inversion of a pyheatmy column (inversion_column.py), moved from pyheatmy/config.py
SIGMA2_MIN_T = 0.001
SIGMA2_MAX_T = 1.0
RANDOMWALKSIGMAT = 0.01
QUANTILE_MIN = 0.05
MEDIANE = 0.5
QUANTILE_MAX = 0.95
N_PARAM_MCMC = 5
GELMANRCRITERIA = 1.2
NBCHAINS = 10
# MCMC parametrization
NITMCMC = 200
NBBURNING = 25
NSAMPLEMIN = 200 #200 is the minimal number of sample for a proper calculation of the quantiles, pb of initialisation
