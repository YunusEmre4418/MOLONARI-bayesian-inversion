"""MCMC estimation of the parameters of any model from its measurements (DREAM with several chains, random walk
Metropolis with one), and its result. The model is a function of the physical parameters, one mapping of parameter
names to values per block of the state space, that returns a mapping of named outputs: for a pyheatmy column,
Column.simulate, whose blocks are the layers. This module does not import the model's package.

The code of the run was moved from pyheatmy.Column.compute_mcmc (then InversionColumn.compute_mcmc), with its
comments; the samplers are in samplers.py."""

import random
import sys
import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Mapping, Sequence, Union

import numpy as np
from tqdm import trange

from bayesian_inversion.config import (
    GELMANRCRITERIA,
    MEDIANE,
    NBBURNING,
    NBCHAINS,
    NITMCMC,
    NSAMPLEMIN,
    QUANTILE_MAX,
    QUANTILE_MIN,
    RANDOMWALKSIGMAT,
    SIGMA2_MAX_T,
    SIGMA2_MIN_T,
)
from bayesian_inversion.diagnostics import gelman_rubin
from bayesian_inversion.energy import compute_energy
from bayesian_inversion.prior import Prior
from bayesian_inversion.samplers import DREAM, RandomWalk
from bayesian_inversion.state import State
from bayesian_inversion.state_space import StateSpace


@dataclass
class MCMCResult:
    """The states of an MCMC run and the quantiles of the stored outputs.

    states: with DREAM, the state of each chain at the end of the burn-in, then one state per chain and per
    iteration, chain by chain; with random walk Metropolis, the start of the chain, then one state per iteration. A
    rejected proposal repeats the previous state of its chain (the same State object).
    acceptance: acceptance rate of each chain (DREAM, an array) or of the chain (random walk Metropolis, a float).
    quantiles: for each stored output, the samples' quantiles, {quantile: array}.
    nb_burn_in_iter: iterations of the DREAM burn-in (None with random walk Metropolis)."""

    state_space: StateSpace
    states: list
    nb_chain: int
    acceptance: Union[np.ndarray, float]
    quantiles: dict
    nb_burn_in_iter: Union[int, None]

    def energies(self):
        return np.array([state.energy for state in self.states])

    def sigma2(self):
        """Physical sigma2 of each state."""
        return np.array([state.sigma2 for state in self.states])

    def params(self):
        """Physical parameters of each state: array (number of states, number of blocks, number of parameters)."""
        return np.array([self.state_space.physical_array(state.params) for state in self.states])

    def chains(self):
        """params() iteration by iteration: array (iterations + 1, chains, blocks, parameters)."""
        return self.params().reshape(-1, self.nb_chain, *self.state_space.shape)

    def best_state(self):
        """The state of lowest energy (the first one on a tie)."""
        return min(self.states, key=lambda state: state.energy)

    def best(self):
        """Physical parameters of the state of lowest energy, one mapping per block: an argument of the model."""
        return self.state_space.to_physical(self.best_state().params)


def run_mcmc(
    model: Callable[[Sequence[Mapping[str, float]]], Mapping[str, np.ndarray]],
    observations: np.ndarray,
    state_space: StateSpace,
    predicted: str = "sensors",
    stored: Sequence[str] = ("temperatures", "flows"),
    quantile: Union[float, Sequence[float]] = (QUANTILE_MIN, MEDIANE, QUANTILE_MAX),
    verbose=False,
    typealgo="no sigma",
    sigma2=1.0,
    nb_iter=NITMCMC,
    nb_chain=NBCHAINS,
    nitmaxburning=NBBURNING,
    delta=3,
    n_CR=3,
    c=0.1,
    c_star=1e-12,
    n_sous_ech_time=1,
    n_sous_ech_space=1,
    threshold=GELMANRCRITERIA,
    seed=None,
):
    """Estimate the parameters of state_space from observations, with DREAM if nb_chain > 1 and random walk
    Metropolis otherwise. model(params) returns a mapping of outputs: output[predicted] is compared with
    observations (Gaussian errors of variance sigma2, NaN observations left out), and the outputs named in stored,
    arrays (cells, times), are subsampled for the quantiles, one cell out of n_sous_ech_space and one time out of
    n_sous_ech_time. typealgo "no sigma" keeps sigma2 fixed; otherwise sigma2 is estimated, with the Jeffreys prior
    on (SIGMA2_MIN_T, SIGMA2_MAX_T). seed, when given, seeds the random generators of Python and numpy, which the
    samplers use, before the run. Returns an MCMCResult."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if verbose:
        print(
            "--- Compute MCMC ---",
            "Priors :",
            f"{state_space}",
            f"Number of iterations : {nb_iter}",
            f"Number of chains : {nb_chain}",
            "--------------------",
            sep="\n",
        )

    n_sous_ech_iter = max(1, int(np.floor(nb_chain * nb_iter / NSAMPLEMIN)))
    sizesubsampling = max(int(np.ceil(nb_iter / n_sous_ech_iter)), 1)

    if verbose:
        print(
            f"Subsampling for Quantile computation every {n_sous_ech_iter} iterations"
        )
        print(
            f"Size of the subsampling per chain : {sizesubsampling} iterations among {nb_iter} iterations"
        )

    if typealgo == "no sigma":  # Dans ce cas là sigma2 est une sorte de Dirac
        sigma2_temp_prior = Prior((sigma2, sigma2), 0, lambda x: 1)
        sigma2_distrib = sigma2_temp_prior.density
    else:
        # uniform in the MCMC space of the prior, log10(sigma2) for this range: the Jeffreys prior 1 / sigma2
        sigma2_temp_prior = Prior((SIGMA2_MIN_T, SIGMA2_MAX_T), RANDOMWALKSIGMAT)
        sigma2_distrib = sigma2_temp_prior.density

    # vérification des types des arguments
    if isinstance(quantile, Number):
        quantile = [quantile]

    states = list()
    nb_burn_in_iter = 0  # nombre d'itération de burn-in

    # the samplers move in the MCMC space of the priors, the model receives physical values
    def sampler_model(x):
        return model(state_space.to_physical(x))

    def energy(output, sigma2):
        return compute_energy(output[predicted], observations, sigma2, sigma2_distrib)

    def subsample(output):
        return {key: output[key][::n_sous_ech_space, ::n_sous_ech_time] for key in stored}

    # float32: these samples only feed the quantiles, and their size grows with the length of the series;
    # allocated at the first sample, whose shapes give the subsampled numbers of cells and times
    samples = {}
    chains = (nb_chain,) if nb_chain > 1 else ()  # DREAM stores one sample per chain

    def store(index, sample):
        for key, values in sample.items():
            if key not in samples:
                samples[key] = np.zeros((sizesubsampling, *chains, *values.shape), np.float32)
            samples[key][index] = values

    # Obligation de faire une disjonction de cas selon le nombre de chaînes, en effet l'algorithme DREAM fonctionne selon une perturbation différentielles entre les
    # différentes chaînes, ce qui ne peut pas être vu comme un cas particulier du Random Walk Metropolis qui est basé lui sur le principe d'une perturbation gaussienne symétrique

    if nb_chain > 1:
        sampler = DREAM(
            sampler_model, energy, state_space.priors, sigma2_temp_prior, nb_chain, delta, n_CR, c, c_star
        )
        is_param_fixed = sampler.is_fixed

        ### initialisation des énergie

        # lancement du modèle direct pour les paramètres tirés des priors, par chaîne
        sampler.initialize()

        ### stockage des résultats
        # XBurnIn va servir à stocker les paramètres pour toutes les itérations de chaque couche de chaque chaîne pendant le Burn In
        # Sert uniquement pour le calcul du critère de Gelman-Rubin

        XBurnIn = np.zeros((nitmaxburning + 1, nb_chain, *state_space.shape), np.float64)

        # en particulier la première instance de XBurnIn est égale à X :

        XBurnIn[0] = sampler.x

        if verbose:
            print("--- Begin Burn in phase ---")

        r_hat = gelman_rubin(XBurnIn[:1], is_param_fixed)
        burn_in_converged = np.all(r_hat[~is_param_fixed] <= threshold)
        for i in trange(nitmaxburning, desc="Burn in phase"):
            sampler.step(adapt=True)

            # Fin d'une itération, on vérifie si on peut sortir du burn-in
            XBurnIn[i + 1] = sampler.x

            r_hat = gelman_rubin(XBurnIn[: i + 2], is_param_fixed)
            burn_in_converged = np.all(r_hat[~is_param_fixed] <= threshold)
            if burn_in_converged:
                if verbose:
                    print(f"Burn-in finished after : {nb_burn_in_iter} iterations")
                break  # on sort du burn-in

            nb_burn_in_iter += 1  # incrémentation du numbre d'itération de burn-in

        if not burn_in_converged:
            l, p = np.unravel_index(
                np.argmax(np.where(is_param_fixed, -np.inf, r_hat)), r_hat.shape
            )
            warnings.warn(
                f"The DREAM burn-in reached nitmaxburning = {nitmaxburning} iterations without converging: "
                f"R_hat = {r_hat[l, p]:.3g} for '{state_space.param_names[p]}' of block '{state_space.block_names[l]}', "
                f"above the threshold {threshold}. The following states may not sample the posterior; "
                f"increase nitmaxburning.",
                UserWarning,
            )

        # L'historique des états commence à la position atteinte en fin de burn-in
        for j in range(nb_chain):
            states.append(
                State(
                    params=sampler.x[j].copy(),  # copy: sampler.x[j] is overwritten in place on acceptance
                    energy=sampler.energy[j],
                    ratio_accept=1,
                    sigma2=sampler.sigma2_physical(j),
                )
            )

        # On suit les taux d'acceptation pour chaque chaîne grâce à un vecteur de taille nb_chain qu'on incrémente à chaque itération
        acceptance = np.zeros(nb_chain, np.float64)

        # Transition après le burn in

        # La variable XBurnIn n'est plus utile

        del XBurnIn

        for i in trange(nb_iter, desc="DREAM MCMC Computation", file=sys.stdout):
            accepted = sampler.step()

            for j in range(nb_chain):
                if accepted[j]:
                    acceptance[j] += 1
                    states.append(
                        State(
                            params=sampler.x[j].copy(), # copy: sampler.x[j] is overwritten in place on acceptance
                            energy=sampler.energy[j],
                            ratio_accept=1,
                            sigma2=sampler.sigma2_physical(j),
                        )
                    )
                else:
                    # On ne met pas à jour l'état :
                    states.append(states[-nb_chain])

            if i % n_sous_ech_iter == 0:  # sous échantillonnage
                # Si le numéro de l'itération i est un multiple de n_sous_ech_iter, on stocke
                k = i // n_sous_ech_iter
                for j in range(nb_chain):
                    sample = subsample(sampler.output[j])  # dernières sorties acceptées
                    if any(np.isnan(values).any() for values in sample.values()):
                        sample = {key: samples[key][(k - 1), j] for key in sample}
                    store((k, j), sample)

        for j in range(nb_chain):
            acceptance[j] /= nb_iter

        # Calcul des taux d'acceptation:
        if verbose == True:
            print(f"Acceptance rate : {acceptance}")

        # Calcul des quantiles
        samples = {key: values.reshape(sizesubsampling * nb_chain, *values.shape[2:]) for key, values in samples.items()}

    else:  # cas single chain
        sampler = RandomWalk(sampler_model, energy, state_space.priors, sigma2_temp_prior)

        for i in trange(nitmaxburning, desc="Init Mcmc ", file=sys.stdout):
            # on tire un jeu de paramètres aléatoires selon les priors, on garde le meilleur
            sampler.draw()

        states = [
            State(
                params=sampler.x[0].copy(),  # copy: sampler.x[0] is overwritten in place on acceptance
                energy=sampler.energy[0],
                ratio_accept=1,
                sigma2=sampler.sigma2_physical(0),
            )
        ]

        nb_accepted = 0

        for i in trange(nb_iter, desc="Mcmc Computation", file=sys.stdout):
            if sampler.step():
                nb_accepted += 1
                states.append(
                    State(
                        params=sampler.x[0].copy(),
                        energy=sampler.energy[0],
                        ratio_accept=nb_accepted / (i + 1),
                        sigma2=sampler.sigma2_physical(0),
                    )
                )

            else:  # le saut est rejeté
                states.append(states[-1])

            if i % n_sous_ech_iter == 0:
                # Si i est un multiple de n_sous_ech_iter, on stocke
                k = i // n_sous_ech_iter
                store((k,), subsample(sampler.output[0]))  # dernières sorties acceptées

        acceptance = nb_accepted / nb_iter
        nb_burn_in_iter = None

        if verbose == True:
            print(f"Acceptance rate : {acceptance}")

    quantiles = {
        key: {quant: res for quant, res in zip(quantile, np.quantile(values, quantile, axis=0))}
        for key, values in samples.items()
    }

    if verbose == True:
        print("Quantiles computed")

    return MCMCResult(state_space, states, nb_chain, acceptance, quantiles, nb_burn_in_iter)
