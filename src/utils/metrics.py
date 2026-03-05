""" Module de tests numériques de convergence des méthodes numériques pour les deux solution"""
import numpy as np
import time
import src.models.fd.solver as slv_fd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import src.models.pinn.trainers as tr
from src.models.pinn.network import PINN
import torch
from src.models.domain import make_masks


def interpolate_on_finer_grid(U,XX_coarse, YY_coarse, XX_fine, YY_fine):
    """
    Interpole la solution U définie sur la grille coarse (XX_coarse, YY_coarse) 
    sur la grille fine (XX_fine, YY_fine) par interpolation bilinéaire
    """
    interpolator = RegularGridInterpolator((YY_coarse[:,0], XX_coarse[0,:]), U)
    points_fine = np.stack([YY_fine.ravel(), XX_fine.ravel()], axis = -1)
    U_fine = interpolator(points_fine).reshape(XX_fine.shape)
    return U_fine

def convergence_fd(params, N_list, f, u_exacte = None): 
    """
    Lance le solveur FD associé a f, renvoie les erreurs de raffinement et le temps de calcul
    """
    errors = []
    errex = []
    times = []
    Hs = []
    U_prev, XX_prev, YY_prev = None, None, None
    for N in N_list: 
        Hs.append((params.X_max - params.X_min)/N)
        t0 = time.time()
        U, XX_cur, YY_cur = slv_fd.Solveur_Zermelo(N, f, params)
        t1 = time.time()
        times.append(t1-t0)
        if u_exacte is not None:
            mask = ~np.isnan(U)
            err = np.nanmax(np.abs(U[mask] - u_exacte(XX_cur[mask], YY_cur[mask], params)))
            errex.append(err)

        if U_prev is not None:
            mask = ~np.isnan(U)
            U_prev_interp = interpolate_on_finer_grid(U_prev, XX_prev, YY_prev, XX_cur, YY_cur)
            error = np.nanmax(np.abs(U[mask] - U_prev_interp[mask]))
            errors.append(error)
            
        U_prev, XX_prev, YY_prev = U, XX_cur, YY_cur

    return errors, times, (errex if u_exacte is not None else None)


def convergence_pinn_epochs(params, f,N_fd = 1000, u_ref=None, epoch_list=None):
    """
    u_ref : callable optionnel.
        Si fourni : erreur vs solution de référence.
        Sinon : erreur vs solution de référence calculée par FD sur une grille fine de N_fd x N_fd points.
    """
    
    ## mode d'évaluation de la solution sur grille régulière

    if u_ref is not None:
        mode = "analytique"
    else:
        mode = "fd"
    
    ## grille d'évaluation de la solution (calculée une seule fois)
    XX = torch.linspace(params.X_min, params.X_max, N_fd)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
    _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp = torch)
    XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
    
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp = torch).flatten()
    else : 
        t0 = time.time()
        print(f"Convergence PINN : calcul de la solution de référence par FD sur une grille de {N_fd}x{N_fd} points...")
        u_fd, XX_np, YY_np = slv_fd.Solveur_Zermelo(N_fd-1, f, params)
        u_ref_vals = torch.tensor(np.nan_to_num(u_fd), dtype=torch.float32).flatten()
        t1 = time.time()
        print(f"Temps de calcul de la solution de référence : {t1-t0:.2f} secondes")
    print("Entrainement du PINN...")
    ## Entrainement incrémental du PINN
    model = PINN()
    Time = []
    L_moy = []
    errors = []
    for i, epochs in enumerate(epoch_list):
        loss, tim = tr.train(model, params, f, n_epochs=epochs-(epoch_list[i-1] if i > 0 else 0))
        Time.append(tim)
        L_moy.append(np.mean(loss))

        with torch.no_grad():
            u_pred = model(XY).squeeze()
        ## erreur L_inf par rapport à la solution de référence (analytique ou FD)
        diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
        err = torch.max(diff[mask_PDE.flatten()])
        errors.append(err.item())

        
    return L_moy, Time, errors, mode
## Paramètres de défaut pour le test suivant
DEFAULTS = {
    "lr":          1e-3,
    "lam":         5.0,
    "hidden_size": 64,
    "n_layers":    4,
    "N_colloc":    200,
    "N_bord":      50,
}

def sensibilité_pinn(params,hyper_params, f, max_epochs  = 10000, u_ref = None, N_fd = 1000): 
    """
    Étudie la sensibilité du PINN à différents hyperparamètres (ex: lr, lam, N_colloc, N_bord).
    Renvoie les erreurs de convergence pour chaque configuration d'hyperparamètres.
    """
    ## référence précalculée en une fois : 
    if u_ref is not None:
        mode = "analytique"
    else:
        mode = "fd"
    
    ## grille d'évaluation de la solution (calculée une seule fois)
    XX = torch.linspace(params.X_min, params.X_max, N_fd)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
    _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp = torch)
    XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
    
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp = torch).flatten()
    else : 
        t0 = time.time()
        print(f"Convergence PINN : calcul de la solution de référence par FD sur une grille de {N_fd}x{N_fd} points...")
        u_fd, XX_np, YY_np = slv_fd.Solveur_Zermelo(N_fd-1, f, params)
        u_ref_vals = torch.tensor(np.nan_to_num(u_fd), dtype=torch.float32).flatten()
        t1 = time.time()
        print(f"Temps de calcul de la solution de référence : {t1-t0:.2f} secondes")
    print("Entrainement du PINN...")

    ## boucle Principale
    results = {}

    for hp_name, hp_values in hyper_params.items():
        print(f" Hyperparamètre : {hp_name}| valeurs : {hp_values}")

        results[hp_name] = {}

        for val in hp_values :
            hp = {**DEFAULTS, hp_name : val}
            print(f"-> {hp_name} = {val} (autres : Défaut)")

            model = PINN(hidden_size=hp["hidden_size"], n_layers= hp["n_layers"])

            t0 = time.time()
            loss_hist, _ = tr.train(model, params, f, 
                                    N_colloc = hp["N_colloc"],
                                    N_bord=hp["N_bord"],
                                    n_epochs = max_epochs,
                                    lam = hp["lam"], 
                                    lr = hp["lr"])
            total_time = time.time() - t0

            #Erreur L infty posterieure a l'entrainement
            with torch.no_grad():
                u_pred = model(XY).squeeze()
             ## erreur L_inf par rapport à la solution de référence (analytique ou FD)
            diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
            err = torch.max(diff[mask_PDE.flatten()])

            results[hp_name][val] = {
                "loss" : loss_hist, 
                "time" : total_time,
                "error" : err,
            }

            print(f"loss_finale = {loss_hist[-1]}"
                  f"err L_infty = {err:.3e} " f"temps = {total_time :.1f}s")
    return results