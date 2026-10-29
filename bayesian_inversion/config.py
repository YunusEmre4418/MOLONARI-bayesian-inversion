# Thresholds of the sampling scale chosen by Prior (linear, log or symlog)
LOG_SCALE_THRESHOLD= 1000**0.5 # Moyenne géométrique entre 1e1 et 1e2
SYMLOG_MAGNITUDE_THRESHOLD = 0.1
SYMLOG_LINTHRESH_RATIO= 1e2

GAMMA_FACTOR = 2.38  # DREAM jump scale: gamma = GAMMA_FACTOR / sqrt(2 * delta * d*)
DREAM_PROB_MODE_JUMP = 0.2   # DREAM: probability of using gamma = DREAM_GAMMA_MODE_JUMP, to jump between posterior modes
DREAM_GAMMA_MODE_JUMP = 1.0
