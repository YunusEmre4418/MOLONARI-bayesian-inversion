import numpy as np

from bayesian_inversion.config import DREAM_GAMMA_MODE_JUMP, DREAM_PROB_MODE_JUMP, GAMMA_FACTOR


def perturbation_DREAM(
    nb_chain,
    nb_layer,
    nb_param,
    X,
    id_layer,
    j,
    delta,
    ncr,
    c,
    c_star,
    cr_vec,
    pcr,
    ranges,
    is_param_fixed,  # <-- NOUVEL ARGUMENT
):
    X_proposal = np.zeros((nb_layer, nb_param), np.float64)
    dX = np.zeros((nb_layer, nb_param), np.float64)

    for l in range(nb_layer):
        # --- LOGIQUE MODIFIÉE POUR IGNORER LES PARAMÈTRES FIXES ---

        # On identifie les paramètres avec lesquels on a le droit de "jouer".
        variable_indices = np.where(is_param_fixed[l] == False)[0]
        n_variable_params = len(variable_indices)

        # Si tous les paramètres de cette couche sont fixes, on ne fait rien.
        if n_variable_params == 0:
            X_proposal[l] = X[j, l]
            continue  # On passe à la couche suivante

        # La logique de DREAM ne s'applique que sur les paramètres variables.
        id_layer[l] = np.random.choice(ncr, p=pcr[l])
        z = np.random.uniform(
            0, 1, n_variable_params
        )  # On ne tire que pour les variables

        # On décide quels paramètres VARIABLES seront perturbés
        A_variable = z <= cr_vec[id_layer[l]]
        d_star = np.sum(A_variable)

        if d_star == 0:  # On s'assure d'en perturber au moins un
            A_variable[np.argmin(z)] = True
            d_star = 1

        # On reconstruit le masque de perturbation complet 'A'
        A = np.zeros(nb_param, dtype=bool)
        jump_indices = variable_indices[A_variable]
        A[jump_indices] = True

        # Le reste du calcul du saut ne change pas, car le masque 'A'
        # sélectionnera automatiquement les bons paramètres.
        lambd = np.random.uniform(-c, c, d_star)
        zeta = np.random.normal(0, c_star, d_star)

        available_indices = np.delete(np.arange(nb_chain), j)
        a = np.random.choice(available_indices, delta, replace=False)
        remaining_indices = np.setdiff1d(available_indices, a)
        b = np.random.choice(remaining_indices, delta, replace=False)
        gamma = GAMMA_FACTOR / np.sqrt(2 * d_star * delta)
        if np.random.uniform() < DREAM_PROB_MODE_JUMP:
            gamma = DREAM_GAMMA_MODE_JUMP

        # Le calcul de dX ne se fait que sur les indices où A est True.
        dX[l][A] = zeta + (1 + lambd) * gamma * np.sum(
            X[a, l][:, A] - X[b, l][:, A], axis=0
        )

        X_proposal[l] = X[j, l] + dX[l]

        # On calcule la largeur de l'intervalle pour tous les paramètres
        width = ranges[l][:, 1] - ranges[l][:, 0]

        # On crée un masque pour identifier les paramètres qui sont réellement variables (width > 0)
        variable_mask = width > 0

        # On applique la condition aux bords UNIQUEMENT aux paramètres variables.
        # Les paramètres fixes (où width=0) ne sont pas touchés par cette ligne.
        X_proposal[l][variable_mask] = ranges[l][variable_mask, 0] + np.mod(
            X_proposal[l][variable_mask] - ranges[l][variable_mask, 0],
            width[variable_mask],
        )

    return (X_proposal, dX, id_layer)
