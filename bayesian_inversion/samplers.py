"""MCMC samplers that do not know the model: DREAM with several chains and random walk Metropolis with one chain.
A sampler moves its chains in the MCMC space of the priors and sees the model only through two functions:

- model(x) -> output: x has one row per block of parameters (a layer of a pyheatmy column) and one column per
  parameter, in the MCMC space of the priors; output is whatever energy needs;
- energy(output, sigma2) -> float: the negative log-posterior of the measurements for this output, with the
  physical variance sigma2 of the measurement errors.

The chains also move sigma2, in the MCMC space of its prior. The caller runs the iterations one step at a time,
records the states and decides when the burn-in ends."""

from random import random

import numpy as np

from bayesian_inversion.dream import perturbation_DREAM
from bayesian_inversion.energy import compute_log_acceptance


class Sampler:
    """Common body of the samplers: the current state of each chain (parameters, sigma2, model output and
    energy), the draws from the priors and the Metropolis acceptance. priors has one sequence of Prior per
    block; sigma2_prior is the Prior of sigma2."""

    def __init__(self, model, energy, priors, sigma2_prior, nb_chain):
        self.model = model
        self.energy_function = energy
        self.priors = priors
        self.sigma2_prior = sigma2_prior
        self.nb_chain = nb_chain
        self.nb_block = len(priors)
        self.nb_param = len(priors[0])
        self.is_fixed = np.array([[prior.is_fixed for prior in block] for block in priors], dtype=bool)
        self.x = np.zeros((nb_chain, self.nb_block, self.nb_param), np.float64)
        self.sigma2 = np.zeros(nb_chain, np.float64)  # in the MCMC space of sigma2_prior
        self.energy = np.zeros(nb_chain, np.float64)
        self.output = [None] * nb_chain  # model output of the current state of each chain

    def sample_prior(self):
        """Parameters drawn from the priors, block by block, in the MCMC space."""
        return np.array([[prior.sample() for prior in block] for block in self.priors], np.float64)

    def sigma2_physical(self, j):
        """Physical sigma2 of chain j."""
        return self.sigma2_prior.mcmc_to_physical(self.sigma2[j])

    def evaluate(self, x, sigma2):
        """Model output for x and its energy, sigma2 in the MCMC space of its prior."""
        output = self.model(x)
        return output, self.energy_function(output, self.sigma2_prior.mcmc_to_physical(sigma2))

    def uniform(self):
        """Uniform draw on [0, 1) of the acceptance test."""
        raise NotImplementedError

    def metropolis(self, j, x_proposal, sigma2_proposal):
        """Evaluate the proposal of chain j and accept it with the Metropolis probability; return whether it was
        accepted."""
        output, energy = self.evaluate(x_proposal, sigma2_proposal)
        if np.log(self.uniform()) < compute_log_acceptance(energy, self.energy[j]):
            self.x[j] = x_proposal
            self.sigma2[j] = sigma2_proposal
            self.energy[j] = energy
            self.output[j] = output
            return True
        return False


class DREAM(Sampler):
    """DREAM (Vrugt et al., 2009; Vrugt, 2016): the proposal of a chain is built from the differences between
    other chains (perturbation_DREAM), block by block, with crossover probabilities adapted during the burn-in."""

    def __init__(self, model, energy, priors, sigma2_prior, nb_chain, delta, n_CR, c, c_star):
        # the proposal of a chain uses two disjoint groups of delta other chains (perturbation_DREAM)
        if nb_chain < 2 * delta + 1:
            raise ValueError(
                f"DREAM needs at least 2 * delta + 1 = {2 * delta + 1} chains for delta = {delta}: the proposal of a "
                f"chain uses two disjoint groups of delta other chains; got nb_chain = {nb_chain}."
            )
        super().__init__(model, energy, priors, sigma2_prior, nb_chain)
        self.delta = delta
        self.n_CR = n_CR
        self.c = c
        self.c_star = c_star
        # bounds of the parameters in the MCMC space: the proposals are wrapped around them
        self.ranges = np.array([[prior.mcmc_range for prior in block] for block in priors], np.float64)
        self.cr_vec = np.arange(1, n_CR + 1) / n_CR  # Valeur des crossover
        self.n_id = np.zeros((self.nb_block, n_CR), np.float64)  # Nombre de fois où chaque crossover est utilisé
        self.J = np.zeros((self.nb_block, n_CR), np.float64)  # Matrice des sauts
        # Probabilité de choisir un crossover, initialisé de manière uniforme
        self.pcr = np.ones((self.nb_block, n_CR)) / n_CR
        # Vecteur qui recense les indices de crossover choisis pour chaque couche
        self.id_layer = np.zeros(self.nb_block, np.int32)

    def uniform(self):
        return np.random.uniform(0, 1)

    def initialize(self):
        """Draw the parameters and sigma2 of each chain from the priors and evaluate them."""
        for j in range(self.nb_chain):
            self.x[j] = self.sample_prior()
            self.sigma2[j] = self.sigma2_prior.sample()
            self.output[j], self.energy[j] = self.evaluate(self.x[j], self.sigma2[j])

    def step(self, adapt=False):
        """One iteration: a proposal for each chain in turn, accepted or not. With adapt (burn-in), the crossover
        probabilities are updated from the jumps. Return whether each chain moved."""
        accepted = np.zeros(self.nb_chain, dtype=bool)
        std_X = np.std(self.x, axis=0)  # calcul des écarts types des paramètres
        # afin d'éviter la division par zéro dans le calcul du saut dans le cas d'un paramètre fixe
        std_X[std_X == 0] = 1.0

        for j in range(self.nb_chain):
            # On tire un nouveau jeu de paramètres x_proposal, les indices de crossover choisis
            # et la perturbation dX
            x_proposal, dX, self.id_layer = perturbation_DREAM(
                self.nb_chain,
                self.nb_block,
                self.nb_param,
                self.x,
                self.id_layer,
                j,
                self.delta,
                self.n_CR,
                self.c,
                self.c_star,
                self.cr_vec,
                self.pcr,
                self.ranges,
                self.is_fixed,
            )
            sigma2_proposal = self.sigma2_prior.perturb(self.sigma2[j])  # On tire un nouveau sigma2
            accepted[j] = self.metropolis(j, x_proposal, sigma2_proposal)

            if adapt:
                if not accepted[j]:
                    dX = np.zeros((self.nb_block, self.nb_param))
                # Mise à jour du vecteur saut pour chaque couche
                for l in range(self.nb_block):
                    self.J[l, self.id_layer[l]] += np.sum((dX[l] / std_X[l]) ** 2)
                    self.n_id[l, self.id_layer[l]] += 1

        if adapt:
            # Mise à jour du pcr pour chaque couche pour DREAM
            for l in range(self.nb_block):
                # On met à jour pcr avec la qualité des sauts J
                used = self.n_id[l] != 0
                self.pcr[l][used] = self.J[l][used] / self.n_id[l][used]

                # On vérifie si la somme des probabilités est nulle
                if np.sum(self.pcr[l]) == 0:
                    # Si oui (ce qui arrive si seul le paramètre fixe a été perturbé),
                    # on réinitialise les probabilités à une distribution uniforme.
                    self.pcr[l][:] = 1.0 / self.n_CR
                else:
                    # Sinon, on normalise comme d'habitude.
                    self.pcr[l] = self.pcr[l] / np.sum(self.pcr[l])

        return accepted


class RandomWalk(Sampler):
    """Random walk Metropolis with one chain: each parameter moves by a Gaussian step of the sigma of its prior,
    wrapped around its range (Prior.perturb). The chain starts from the best of several draws from the priors."""

    def __init__(self, model, energy, priors, sigma2_prior):
        super().__init__(model, energy, priors, sigma2_prior, nb_chain=1)
        self.nb_draws = 0

    def uniform(self):
        return random()

    def draw(self):
        """Draw the parameters and sigma2 from the priors and keep them as the start of the chain if their energy
        is the lowest so far (the first draw on a tie)."""
        x = self.sample_prior()
        sigma2 = self.sigma2_prior.sample()
        output, energy = self.evaluate(x, sigma2)
        if self.nb_draws == 0 or energy < self.energy[0]:
            self.x[0], self.sigma2[0], self.energy[0], self.output[0] = x, sigma2, energy, output
        self.nb_draws += 1

    def step(self):
        """One iteration: a proposal, accepted or not. Return whether the chain moved."""
        x_proposal = np.array(
            [
                [prior.perturb(value) for prior, value in zip(block, values)]
                for block, values in zip(self.priors, self.x[0])
            ],
            np.float64,
        )
        sigma2_proposal = self.sigma2_prior.perturb(self.sigma2[0])
        return self.metropolis(0, x_proposal, sigma2_proposal)
