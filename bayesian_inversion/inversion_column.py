"""Inversion of a pyheatmy column: InversionColumn is a pyheatmy Column with the MCMC estimation of its layer
parameters (DREAM with several chains, random walk Metropolis with one), its records, getters and plots.
The methods were moved from pyheatmy.Column as they were."""

from typing import Sequence, Union
from random import random, choice
from operator import attrgetter
from numbers import Number
import sys
import warnings
import psutil
import types

import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import trange

from pyheatmy import Column
from pyheatmy.checker import checker
from pyheatmy.core import rmse_measured
from pyheatmy.params import Param, PARAM_LIST

from bayesian_inversion.config import (
    GELMANRCRITERIA,
    MEDIANE,
    N_PARAM_MCMC,
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
from bayesian_inversion.dream import perturbation_DREAM
from bayesian_inversion.energy import compute_energy, compute_log_acceptance
from bayesian_inversion.prior import Prior
from bayesian_inversion.state import State


class InversionColumn(Column):
    """A pyheatmy Column whose layer parameters are estimated by MCMC from the temperature measurements."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # liste contenant des objets de classe état et de longueur le nombre d'états acceptés par la MCMC (<=nb_iter), passe à un moment par une longueur de 1000 pendant l'initialisation de MCMC
        self._states = list()
        # dictionnaire indexé par les quantiles (0.05,0.5,0.95) à qui on a associe un array de deux dimensions : dimension 1 les profondeurs, dimension 2 : liste des valeurs de températures associées au quantile, de longueur les temps de mesure
        self._quantiles_temperatures = None
        # dictionnaire indexé par les quantiles (0.05,0.5,0.95) à qui on a associe un array de deux dimensions : dimension 1 les profondeurs, dimension 2 : liste des valeurs de débits spécifiques associés au quantile, de longueur les temps de mesure
        self._quantiles_flows = None

    @checker
    def compute_mcmc(
        self,
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
    ):
        if verbose:
            print(
                "--- Compute MCMC ---",
                "Priors :",
                *(
                    f"    {prior}"
                    for prior in [layer.Prior_list for layer in self.all_layers]
                ),
                f"Number of cells : {self._nb_cells}",
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

        process = psutil.Process()

        if typealgo == "no sigma":  # Dans ce cas là sigma2 est une sorte de Dirac
            sigma2_temp_prior = Prior((sigma2, sigma2), 0, lambda x: 1)
            sigma2_distrib = sigma2_temp_prior.density
        else:
            # uniform in the MCMC space of the prior, log10(sigma2) for this range: the Jeffreys prior 1 / sigma2
            sigma2_temp_prior = Prior((SIGMA2_MIN_T, SIGMA2_MAX_T), RANDOMWALKSIGMAT)
            sigma2_distrib = sigma2_temp_prior.density
        # the chains move sigma2 in the MCMC space of its prior; the energy and the states use the physical value
        sigma2_physical = sigma2_temp_prior.mcmc_to_physical

        # vérification des types des arguments
        if isinstance(quantile, Number):
            quantile = [quantile]

        # définition des paramètres de la simulation
        dz = self._real_z[-1] / self._nb_cells
        _z_solve = dz / 2 + np.array([k * dz for k in range(self._nb_cells)])
        ind_ref = [np.argmin(np.abs(z - _z_solve)) for z in self._real_z[1:-1]]
        temp_ref = self._T_measures[:, :].T
        self._states = list()

        # quantités des différents paramètres
        nb_layer = len(self.all_layers)  # nombre de couches
        nb_param = N_PARAM_MCMC  # nombre de paramètres à estimer par couche
        nb_accepted = 0  # nombre de propositions acceptées
        nb_burn_in_iter = 0  # nombre d'itération de burn-in

        is_param_fixed = np.zeros((nb_layer, nb_param), dtype=bool)
        for l, layer in enumerate(self.all_layers):
            for p, prior in enumerate(layer.Prior_list):
                if prior.is_fixed:
                    is_param_fixed[l, p] = True

        # variables pour l'état courant
        temp_proposal = np.zeros((self._nb_cells, len(self._times)), np.float64)

        # variables liées au sous-échantillonnage
        nb_cells_sous_ech = int(np.ceil(self._nb_cells / n_sous_ech_space))
        nb_times_sous_ech = int(np.ceil(len(self._times) / n_sous_ech_time))

        # Obligation de faire une disjonction de cas selon le nombre de chaînes, en effet l'algorithme DREAM fonctionne selon une perturbation différentielles entre les
        # différentes chaînes, ce qui ne peut pas être vu comme un cas particulier du Random Walk Metropolis qui est basé lui sur le principe d'une perturbation gaussienne symétrique

        if nb_chain > 1:
            # Voir si ces variables ne peuvent pas devenir des attributs de State
            _temp_iter_chain = np.zeros(
                (nb_chain, self._nb_cells, len(self._times)), np.float64
            )  # dernière température acceptée pour chaque chaine
            _flow_iter_chain = np.zeros(
                (nb_chain, self._nb_cells, len(self._times)), np.float64
            )  # dernier débit accepté pour chaque chaine

            # nombre d'itérations sous-échantillonnées avec initialisation
            # float32: these samples only feed the quantiles, and their size grows with the length of the series
            _temp = np.zeros(
                (sizesubsampling, nb_chain, nb_cells_sous_ech, nb_times_sous_ech),
                np.float32,
            )  # stockage des températures sous échantillonées pendant la mcmc
            _flows = np.zeros(
                (sizesubsampling, nb_chain, nb_cells_sous_ech, nb_times_sous_ech),
                np.float32,
            )  # stockage des débits sous échantillonées pendant la mcmc

            # création de la matrice des bornes des paramètres (sert à s'assurer que la proposition de paramètres est dans les bornes)
            ranges = np.empty((nb_layer, nb_param, 2))
            for l in range(nb_layer):
                for p in range(nb_param):
                    lower_bound, upper_bound = (
                        self.all_layers[l].Prior_list[p].mcmc_range
                    )
                    ranges[l, p] = [
                        lower_bound,
                        upper_bound,
                    ]  # On prend les priors du paramètre p de la couche l.

            # objets liés à DREAM
            cr_vec = np.arange(1, n_CR + 1) / n_CR  # Valeur des crossover
            n_id = np.zeros(
                (nb_layer, n_CR), np.float64
            )  # Nombre de fois où chaque crossover est utilisé
            J = np.zeros((nb_layer, n_CR), np.float64)  # Matrice des sauts
            pcr = (
                np.ones((nb_layer, n_CR)) / n_CR
            )  # Probabilité de choisir un crossover, initialisé de manière uniforme
            id_layer = np.zeros(
                nb_layer, np.int32
            )  # Vecteur qui recense les indices de crossover choisis pour chaque couche

            # Création d'autant d'instance de colonnes que de chaines

            multi_chain = [copy.deepcopy(self) for _ in range(nb_chain)]

            # On redéfinit l'appel du modèle direct sur les deep copies sinon le modèle direct s'exécute sur les paramètre de la colonne self
            # Raisons inconnues peut-être liées au décorateur @checker

            for column in multi_chain:
                column.compute_solve_transi = types.MethodType(
                    Column.compute_solve_transi, column
                )

            ### initialisation des énergie

            # lancement du modèle direct et mise à jours de tous les paramètres dans la matrice X, les énergies dans la matrice Energy
            # et on les stocke dans l'attribut _state de la colonne qui ne sert qu'à plot les distribution et à récupérer les paramètres qui minimisent l'énergie

            self._states = list()
            X = np.zeros((nb_chain, nb_layer, nb_param), np.float64)
            Energy = np.zeros((nb_chain), np.float64)
            sigma2_chain = np.zeros(nb_chain, np.float64)  # sigma2 courant de chaque chaîne

            for j, column in enumerate(multi_chain):
                column.sample_params_from_priors()
                X[j] = column._get_list_mcmc_params()
                sigma2_chain[j] = sigma2_temp_prior.sample()

                column.compute_solve_transi(verbose=False)
                _temp_iter_chain[j] = column.get_temperatures_solve()
                _flow_iter_chain[j] = column.get_flows_solve()
                Energy[j] = compute_energy(
                    _temp_iter_chain[j][ind_ref], temp_ref, sigma2_physical(sigma2_chain[j]), sigma2_distrib
                )

            ### stockage des résultats
            # XBurnIn va servir à stocker les paramètres pour toutes les itérations de chaque couche de chaque chaîne pendant le Burn In
            # Sert uniquement pour le calcul du critère de Gelman-Rubin

            XBurnIn = np.zeros(
                (nitmaxburning + 1, nb_chain, nb_layer, nb_param), np.float64
            )

            # en particulier la première instance de XBurnIn est égale à X :

            XBurnIn[0] = X

            print(
                f"Initialisation - Utilisation de la mémoire (en Mo) : {process.memory_info().rss / 1e6}"
            )

            if verbose:
                print("--- Begin Burn in phase ---")

            r_hat = gelman_rubin(XBurnIn[:1], is_param_fixed)
            burn_in_converged = np.all(r_hat[~is_param_fixed] <= threshold)
            for i in trange(nitmaxburning, desc="Burn in phase"):
                # Initialisation pour les nouveaux paramètres
                std_X = np.std(X, axis=0)  # calcul des écarts types des paramètres
                # afin d'éviter la division par zéro dans le calcul du saut dans le cas d'un paramètre fixe
                std_X[std_X == 0] = 1.0

                for j, column in enumerate(multi_chain):
                    # On lance la perturbation DREAM pour cette colonne, on en tire un nouveau jeu de paramètre X_proposal,
                    # l'ensemble des indices de crossover choisis et la perturbation dX_colonne
                    X_proposal, dX, id_layer = perturbation_DREAM(
                        nb_chain,
                        nb_layer,
                        nb_param,
                        X,
                        id_layer,
                        j,
                        delta,
                        n_CR,
                        c,
                        c_star,
                        cr_vec,
                        pcr,
                        ranges,
                        is_param_fixed,
                    )
                    sigma2_temp_proposal = sigma2_temp_prior.perturb(
                        sigma2_chain[j]
                    )  # On tire un nouveau sigma2

                    # On met à jour les paramètres de la colonne j selon X_proposal pour appeler le modèle direct et calculer l'énergie:
                    for l, layer in enumerate(column.all_layers):
                        layer.mcmc_params = Param(*X_proposal[l])

                    # Calcul du profil de température associé aux nouveaux paramètres
                    column.compute_solve_transi(verbose=False)

                    # On récupère les températures et les débits associés aux nouveaux paramètres
                    temp_proposal = column._temperatures

                    Energy_Proposal = compute_energy(
                        temp_proposal[ind_ref],
                        temp_ref,
                        sigma2_physical(sigma2_temp_proposal),
                        sigma2_distrib,
                    )  # calcul de l'énergie

                    # calcul de la probabilité d'accpetation
                    log_ratio_accept = Energy[j] - Energy_Proposal

                    # Acceptation ou non des nouveaux paramètres

                    if (
                        np.log(np.random.uniform(0, 1)) < log_ratio_accept
                    ):  # La perturbation est acceptée
                        # on met à jour l'état de la colonne
                        X[j] = (
                            X_proposal  # actualisation des paramètres pour la chaine j
                        )
                        Energy[j] = Energy_Proposal
                        sigma2_chain[j] = sigma2_temp_proposal
                        _temp_iter_chain[j] = (
                            temp_proposal  # actualisation du profil de température
                        )

                        _flow_iter_chain[j] = column.get_flows_solve()  # actualisation du débit


                    else:
                        dX = np.zeros((nb_layer, nb_param))
                        # On remet les anciens paramètres pour la colonne :
                        for l, layer in enumerate(column.all_layers):
                            layer.mcmc_params = Param(*X[j][l])

                    # Mise à jour du vecteur saut pour chaque couche
                    for l in range(nb_layer):
                        J[l, id_layer[l]] += np.sum((dX[l] / std_X[l]) ** 2)
                        n_id[l, id_layer[l]] += 1

                # Mise à jour du pcr pour chaque couche pour DREAM
                for l in range(nb_layer):
                    # On met à jour pcr avec la qualité des sauts J
                    pcr[l][n_id[l] != 0] = J[l][n_id[l] != 0] / n_id[l][n_id[l] != 0]

                    # On vérifie si la somme des probabilités est nulle
                    if np.sum(pcr[l]) == 0:
                        # Si oui (ce qui arrive si seul le paramètre fixe a été perturbé),
                        # on réinitialise les probabilités à une distribution uniforme.
                        pcr[l][:] = 1.0 / n_CR
                    else:
                        # Sinon, on normalise comme d'habitude.
                        pcr[l] = pcr[l] / np.sum(pcr[l])

                # Fin d'une itération, on vérifie si on peut sortir du burn-in
                XBurnIn[i + 1] = X

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
                    f"R_hat = {r_hat[l, p]:.3g} for '{PARAM_LIST[p]}' of layer '{self.all_layers[l].name}', "
                    f"above the threshold {threshold}. The following states may not sample the posterior; "
                    f"increase nitmaxburning.",
                    UserWarning,
                )

            self.nb_burn_in_iter = nb_burn_in_iter

            # L'historique des états commence à la position atteinte en fin de burn-in
            for j in range(nb_chain):
                self._states.append(
                    State(
                        layers=X[j].copy(),  # copy: X[j] is overwritten in place on acceptance
                        energy=Energy[j],
                        ratio_accept=1,
                        sigma2_temp=sigma2_physical(sigma2_chain[j]),
                    )
                )

            # On suit les taux d'acceptation pour chaque chaîne grâce à un vecteur de taille nb_chain qu'on incrémente à chaque itération
            self._acceptance = np.zeros(nb_chain, np.float64)

            # Transition après le burn in

            # La variable XBurnIn n'est plus utile

            del XBurnIn

            print(
                f"Initialisation post burn-in - Utilisation de la mémoire (en Mo) : {process.memory_info().rss / 1e6}"
            )

            for i in trange(nb_iter, desc="DREAM MCMC Computation", file=sys.stdout):
                # Initialisation pour les nouveaux paramètres
                std_X = np.std(X, axis=0)  # calcul des écarts types des paramètres
                std_X[std_X == 0] = (
                    1.0  # afin d'éviter la division par zéro dans le calcul du saut dans le cas d'un paramètre fixe
                )

                for j, column in enumerate(multi_chain):
                    X_proposal, dX, id_layer = perturbation_DREAM(
                        nb_chain,
                        nb_layer,
                        nb_param,
                        X,
                        id_layer,
                        j,
                        delta,
                        n_CR,
                        c,
                        c_star,
                        cr_vec,
                        pcr,
                        ranges,
                        is_param_fixed,
                    )
                    sigma2_temp_proposal = sigma2_temp_prior.perturb(
                        sigma2_chain[j]
                    )  # On tire un nouveau sigma2

                    # Mise à jour des paramètres de la colonne j :
                    for l, layer in enumerate(column.all_layers):
                        layer.mcmc_params = Param(*X_proposal[l])

                    # Calcul du profil de température associé aux nouveaux paramètres
                    column.compute_solve_transi(verbose=False)
                    temp_proposal = (
                        column.get_temperatures_solve()
                    )  # récupération du profil de température

                    Energy_Proposal = compute_energy(
                        temp_proposal[ind_ref],
                        temp_ref,
                        sigma2_physical(sigma2_temp_proposal),
                        sigma2_distrib,
                    )  # calcul de l'énergie
                    # calcul de la probabilité d'accpetation
                    log_ratio_accept = compute_log_acceptance(
                        Energy_Proposal, Energy[j]
                    )

                    # Acceptation ou non des nouveaux paramètres
                    if np.log(np.random.uniform(0, 1)) < log_ratio_accept:
                        # on met à jour l'état de la colonne
                        X[j] = X_proposal
                        Energy[j] = Energy_Proposal
                        sigma2_chain[j] = sigma2_temp_proposal
                        self._acceptance[j] += 1

                        _temp_iter_chain[j] = temp_proposal
                        _flow_iter_chain[j] = column.get_flows_solve()

                        self._states.append(
                            State(
                                layers=X[j].copy(), # copy: X[j] is overwritten in place on acceptance
                                energy=Energy[j],
                                ratio_accept=1,
                                sigma2_temp=sigma2_physical(sigma2_temp_proposal),
                            )
                        )

                    else:
                        dX = np.zeros((nb_layer, nb_param), np.float64)
                        # On ne met pas à jour l'état :
                        self._states.append(self._states[-nb_chain])
                        # On remet les anciens paramètres pour la colonne :
                        for l, layer in enumerate(column.all_layers):
                            layer.mcmc_params = Param(*X[j][l])

                if i % n_sous_ech_iter == 0:  # sous échantillonnage
                    # Si le numéro de l'itération i est un multiple de n_sous_ech_iter, on stocke
                    k = i // n_sous_ech_iter
                    for j in range(nb_chain):
                        if np.isnan(
                            _temp_iter_chain[j, ::n_sous_ech_space, ::n_sous_ech_time]
                        ).any():
                            _temp[k, j] = _temp[(k - 1), j]
                            _flows[k, j] = _flows[(k - 1), j]
                        else:
                            _temp[k, j] = _temp_iter_chain[
                                j, ::n_sous_ech_space, ::n_sous_ech_time
                            ]
                            _flows[k, j] = _flow_iter_chain[
                                j, ::n_sous_ech_space, ::n_sous_ech_time
                            ]

            for j in range(nb_chain):
                self._acceptance[j] /= nb_iter

            # Calcul des taux d'acceptation:
            if verbose == True:
                print(f"Acceptance rate : {self._acceptance}")

            # Calcul des quantiles pour la température
            _temp = _temp.reshape(
                sizesubsampling * nb_chain, nb_cells_sous_ech, nb_times_sous_ech
            )
            _flows = _flows.reshape(
                sizesubsampling * nb_chain, nb_cells_sous_ech, nb_times_sous_ech
            )

        else:  # cas single chain
            _temp_iter = np.zeros(
                (self._nb_cells, len(self._times)), np.float64
            )  # dernière température acceptée pour la colonne
            _flow_iter = np.zeros(
                (self._nb_cells, len(self._times)), np.float64
            )  # dernier débit accepté pour la colonne
            # float32: these samples only feed the quantiles, and their size grows with the length of the series
            _temp = np.zeros(
                (sizesubsampling, self._nb_cells, len(self._times)), np.float32
            )  # stockage des températures sous échantillonées pendant la mcmc
            _flows = np.zeros(
                (sizesubsampling, self._nb_cells, len(self._times)), np.float32
            )  # stockage des débits sous échantillonées pendant la mcmc

            for i in trange(nitmaxburning, desc="Init Mcmc ", file=sys.stdout):
                # on tire un jeu de paramètres aléatoires selon les priors
                self.sample_params_from_priors()
                init_sigma2_temp = sigma2_temp_prior.sample()

                # on lance le modèle direct et on stocke les résultats dans la classe state
                self.compute_solve_transi(verbose=False)
                self._states.append(
                    State(
                        layers=self._get_list_mcmc_params(),
                        energy=compute_energy(
                            self.temperatures_solve[ind_ref, :],
                            temp_ref,
                            sigma2=sigma2_physical(init_sigma2_temp),
                            sigma2_distrib=sigma2_distrib,
                        ),
                        ratio_accept=1,
                        sigma2_temp=sigma2_physical(init_sigma2_temp),
                    )
                )

            # on garde uniquement le meilleur état
            self._states = [min(self._states, key=attrgetter("energy"))]

            # on initialise la colonnne pour les paramètres qui minimisent l'énergie
            for l, layer in enumerate(self.all_layers):
                layer.mcmc_params = Param(*self._states[0].layers[l])

            self.compute_solve_transi(verbose=False)

            _temp_iter = self.get_temperatures_solve()
            _flow_iter = self.get_flows_solve()

            nb_accepted = 0

            for i in trange(nb_iter, desc="Mcmc Computation", file=sys.stdout):
                # on stocke les paramètres avant la proposition de pas
                X = self._get_list_mcmc_params()

                # on fait une proposition de pas
                self.perturb_params()
                sigma2_temp_proposal = sigma2_temp_prior.perturb(
                    sigma2_temp_prior.physical_to_mcmc(self._states[-1].sigma2_temp)
                )

                # on calcule l'énergie pour les nouveaux paramètres
                self.compute_solve_transi(verbose=False)

                Energy_Proposal = compute_energy(
                    self.temperatures_solve[ind_ref, :],
                    temp_ref,
                    sigma2_physical(sigma2_temp_proposal),
                    sigma2_distrib=sigma2_temp_prior.density,
                )

                log_ratio_accept = compute_log_acceptance(
                    Energy_Proposal, self._states[-1].energy
                )

                if np.log(random()) < log_ratio_accept:
                    nb_accepted += 1

                    _temp_iter = self.get_temperatures_solve()
                    _flow_iter = self.get_flows_solve()

                    self._states.append(
                        State(
                            layers=self._get_list_mcmc_params(),
                            energy=Energy_Proposal,
                            ratio_accept=nb_accepted / (i + 1),
                            sigma2_temp=sigma2_physical(sigma2_temp_proposal),
                        )
                    )

                else:  # le saut est rejeté
                    self._states.append(self._states[-1])

                    # On remet les paramètres précédent pour la colonne
                    for l, layer in enumerate(self.all_layers):
                        layer.mcmc_params = Param(*X[l])

                if i % n_sous_ech_iter == 0:
                    # Si i est un multiple de n_sous_ech_iter, on stocke
                    k = i // n_sous_ech_iter
                    _temp[k] = _temp_iter[::n_sous_ech_space, ::n_sous_ech_time]
                    _flows[k] = _flow_iter[::n_sous_ech_space, ::n_sous_ech_time]

            self._acceptance = nb_accepted / nb_iter

            if verbose == True:
                print(f"Acceptance rate : {self._acceptance}")
                print(
                    f"Fin itérations MCMC, avant le calcul des quantiles - Utilisation de la mémoire (en Mo) : {process.memory_info().rss / 1e6}"
                )

        self._quantiles_temperatures = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_temp, quantile, axis=0))
        }

        self._quantiles_flows = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_flows, quantile, axis=0))
        }

        if verbose == True:
            print("Quantiles computed")

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def get_depths_mcmc(self):
        return self._z_solve  # NF 15/9/2022 only used in MolonaviZ where we want all cell coordinates, as stated in the API. Never used in pyheatme so no bug following the change

    depths_mcmc = property(get_depths_mcmc)

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def get_times_mcmc(self):
        return self._times

    times_mcmc = property(get_times_mcmc)

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def sample_param(self):
        # retourne aléatoirement un des couples de paramètres parlesquels est passé la MCMC
        return choice(self.get_all_params())

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def get_best_param(self):
        """return the params that minimize the energy"""
        i_best = min(range(len(self._states)), key=lambda i: self._states[i].energy)
        return self.get_all_params()[i_best] # retourne le couple de paramètres minimisant l'énergie par lequels est passé la MCMC

    @compute_mcmc.needed
    def get_best_sigma2(self):
        """return the best sigma that minimizes the energy"""
        return min(self._states, key=attrgetter("energy")).sigma2_temp

    @compute_mcmc.needed
    def get_best_layers(self):
        """set the params of the columns to those which minimize the energy"""
        """works independently of the type of MCMC"""
        best_layers = min(self._states, key=attrgetter("energy")).layers
        for l, layer in enumerate(self.all_layers):
            layer.mcmc_params = Param(*best_layers[l])

    def get_all_params(self):
        """
        Retourne l'historique complet des paramètres PHYSIQUES pour chaque couche.
        Cette version corrigée sait comment traduire l'historique des valeurs MCMC.
        """
        all_physical_params_history = []
        # self.all_layers contient les vrais objets Layer avec leurs Priors (les "traducteurs")
        layers_with_priors = self.all_layers

        # self._states contient l'historique des valeurs de la MCMC (des tableaux NumPy)
        for state in self._states:
            # state.layers est un tableau NumPy de la forme (nb_couches, nb_parametres)
            mcmc_params_for_state = state.layers
            physical_state_params = []

            # On parcourt chaque couche du modèle
            for l, layer_obj in enumerate(layers_with_priors):
                # On récupère les paramètres MCMC pour cette couche à cet état de la chaîne
                mcmc_params_for_layer = mcmc_params_for_state[l]

                # On utilise les Priors de la couche pour traduire les valeurs MCMC en valeurs physiques
                physical_vals = [
                    prior.mcmc_to_physical(val)
                    for prior, val in zip(layer_obj.Prior_list, mcmc_params_for_layer)
                ]
                physical_state_params.append(Param(*physical_vals))

            all_physical_params_history.append(physical_state_params)
        return all_physical_params_history

    # Les autres getters deviennent beaucoup plus simples car ils utilisent la fonction ci-dessus
    def get_all_IntrinK(self):
        """Retourne l'historique des valeurs PHYSIQUES de IntrinK."""
        all_params = self.get_all_params()
        return [[p.IntrinK for p in state_params] for state_params in all_params]

    def get_all_n(self):
        all_params = self.get_all_params()
        return [[p.n for p in state_params] for state_params in all_params]

    def get_all_lambda_s(self):
        all_params = self.get_all_params()
        return [[p.lambda_s for p in state_params] for state_params in all_params]

    def get_all_rhos_cs(self):
        all_params = self.get_all_params()
        return [[p.rhos_cs for p in state_params] for state_params in all_params]

    def get_all_q_s(self):
        all_params = self.get_all_params()
        return [[p.q_s for p in state_params] for state_params in all_params]

    # Les propriétés pointent vers les nouvelles méthodes
    all_IntrinK = property(get_all_IntrinK)
    all_n = property(get_all_n)
    all_lambda_s = property(get_all_lambda_s)
    all_rhos_cs = property(get_all_rhos_cs)
    all_q_s = property(get_all_q_s)

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def get_all_sigma2(self):
        return [state.sigma2_temp for state in self._states]

    all_sigma = property(get_all_sigma2)

    @compute_mcmc.needed
    def get_all_energy(self):
        return [state.energy for state in self._states]

    all_energy = property(get_all_energy)

    @compute_mcmc.needed
    # retourne toutes les valeurs d'acceptance empirique par lesquels est passée la MCMC
    def get_all_acceptance_ratio(self):
        return self._acceptance

    all_acceptance_ratio = property(get_all_acceptance_ratio)

    @compute_mcmc.needed
    def get_quantiles(self):
        return self._quantiles_temperatures.keys()

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def get_temperatures_quantile(self, quantile):
        return self._quantiles_temperatures[quantile]
        # retourne les valeurs des températures en fonction du temps selon le quantile demandé

    # erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    @compute_mcmc.needed
    def get_flows_quantile(self, quantile):
        return self._quantiles_flows[quantile]

    @compute_mcmc.needed
    def get_RMSE_quantile(self, quantile):
        # Array of RMSE for each sensor (except boundary conditions : river and aquifer), over its measured times
        list_RMSE = np.array(
            [
                rmse_measured(self.get_temperatures_quantile(quantile)[id, :], temperatures_obs)
                for id, temperatures_obs in zip(
                    self.get_id_sensors(), self._T_measures.T
                )
            ]
        )

        # Total RMSE, over the sensors with measurements
        total_RMSE = np.sqrt(np.mean(list_RMSE[~np.isnan(list_RMSE)] ** 2))

        return np.append(list_RMSE, total_RMSE)

    @compute_mcmc.needed
    def plot_quantile_temperatures_sensors(self, tunits="K", fontsize=15):
        zeroT = self.set_zeroT(tunits)
        temps_en_jours = self.create_time_in_day()

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        axes[0].set_ylabel(f"Temperature in {tunits}")

        for i, id in enumerate(self.get_id_sensors()):
            axes[i].set_xlabel("Time in days")
            axes[i].plot(
                temps_en_jours,
                self._T_measures[:, i] - zeroT,
                label="Measurement",
            )
            for q_s in self.get_quantiles():
                axes[i].plot(
                    temps_en_jours,
                    self.get_temperatures_quantile(q_s)[id] - zeroT,
                    label=f"Quantile {q_s}",
                )
            axes[i].legend()
            axes[i].set_title(f"Sensor {i + 1}")

        plt.subplots_adjust(wspace=0.05)

    @compute_mcmc.needed
    def plot_all_param_pdf(self):
        """
        Affiche les fonctions de densité de probabilité (PDF) a posteriori pour chaque
        paramètre de chaque couche, sous forme d'histogrammes.
        Version finale corrigée.
        """
        nb_layers = len(self.all_layers)
        nb_params = len(PARAM_LIST)

        all_params_history_list = self.get_all_params()

        if not all_params_history_list:
            print("Aucune donnée MCMC à afficher.")
            return

        # On convertit la liste en un tableau NumPy 3D pour une manipulation facile.
        all_params_array = np.array(all_params_history_list)

        # CORRECTION : On simplifie la création des axes.
        # L'argument squeeze=False garantit que 'axes' est TOUJOURS un tableau 2D,
        # même s'il n'y a qu'une seule couche (forme (1, 5)).
        fig, axes = plt.subplots(
            nrows=nb_layers,
            ncols=nb_params,
            figsize=(15, 3 * nb_layers if nb_layers > 1 else 5),
            squeeze=False,
        )
        fig.suptitle("Distribution a posteriori des paramètres (PDF)", fontsize=16)

        # On supprime le bloc 'if nb_layers == 1:' qui était la source de l'erreur.

        for id_l in range(nb_layers):
            # On extrait l'historique de tous les paramètres pour cette couche.
            layer_distribs = all_params_array[:, id_l, :]

            # Le reste du code est maintenant correct.
            axes[id_l, 0].hist(
                layer_distribs[:, 0]
            )  # On utilise [:, 0] qui est plus standard
            axes[id_l, 0].set_title(f"Couche {id_l + 1} : IntrinK")

            axes[id_l, 1].hist(layer_distribs[:, 1])
            axes[id_l, 1].set_title(f"Couche {id_l + 1} : n")

            axes[id_l, 2].hist(layer_distribs[:, 2])
            axes[id_l, 2].set_title(f"Couche {id_l + 1} : lambda_s")

            axes[id_l, 3].hist(layer_distribs[:, 3])
            axes[id_l, 3].set_title(f"Couche {id_l + 1} : rhos_cs")

            axes[id_l, 4].hist(layer_distribs[:, 4])
            axes[id_l, 4].set_title(f"Couche {id_l + 1} : q_s")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_darcy_flow_quantile(self):
        temps_en_jours = self.create_time_in_day()

        fig, axes = plt.subplots(
            1, 3, figsize=(20, 10), sharex="col", sharey="row", constrained_layout=True
        )
        axes[0].set_ylabel("Débit en m/s")

        # Store the image objects to use for the color bar
        im_list = []
        for i, q in enumerate(self.get_quantiles()):
            im = axes[i].imshow(
                self.get_flows_quantile(q),
                aspect="auto",
                cmap="Spectral_r",
                extent=[0, temps_en_jours[-1], self._real_z[-1], self._real_z[0]],
            )
            axes[i].set_title(f"Darcy flow quantile : {100 * q} %")
            axes[i].set_xlabel("Time in days)")
            im_list.append(im)

        # Add a common color bar
        cbar = fig.colorbar(
            im_list[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04
        )
        cbar.set_label("Flow Rate (m/s)")

        plt.show()

    def sample_params_from_priors(self):
        # each layer samples its own parameters from its priors (Layer.sample)
        for layer in self.all_layers:
            layer.sample()

    def _get_list_mcmc_params(self):
        """
        Retourne les paramètres de TRAVAIL (MCMC) actuels de chaque couche.
        Cette méthode est destinée à un usage INTERNE par l'algorithme MCMC.
        """
        return [layer.mcmc_params for layer in self.all_layers]

    def perturb_params(self):
        # each layer perturbs its own parameters within its priors (Layer.perturb)
        for layer in self.all_layers:
            layer.perturb()
