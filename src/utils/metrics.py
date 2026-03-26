""" Module de tests numériques de convergence des méthodes numériques pour les deux solution"""
import numpy as np
import os
import time
import src.models.fd.solver as slv_fd
from config.parameters import Params
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import src.models.pinn.trainers as tr
from src.models.pinn.network import PINN
import torch
from src.models.domain import make_masks
import matplotlib.pyplot as plt
import copy
from src.models.pinn.loss import Loss

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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if u_ref is not None:
        mode = "analytique"
    else:
        mode = "fd"
    
    ## grille d'évaluation de la solution (calculée une seule fois)
    XX = torch.linspace(params.X_min, params.X_max, N_fd).to(device)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd).to(device)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
    _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp = torch)
    XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
    
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp = torch).flatten()
    else : 
        t0 = time.time()
        print(f"Convergence PINN : calcul de la solution de référence par FD sur une grille de {N_fd}x{N_fd} points...")
        u_fd, XX_np, YY_np = slv_fd.Solveur_Zermelo(N_fd-1, f, params)
        u_ref_vals = torch.tensor(np.nan_to_num(u_fd).T.flatten(), dtype=torch.float32, device=device)
        t1 = time.time()
        print(f"Temps de calcul de la solution de référence : {t1-t0:.2f} secondes")
    print("Entrainement du PINN...")
    ## Entrainement incrémental du PINN
    model = PINN()
    Time = []
    L_moy = []
    errors = []
    for i, epochs in enumerate(epoch_list):
        loss, tim = tr.train_SGD(model, params, f, n_epochs=epochs-(epoch_list[i-1] if i > 0 else 0))
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
    "N_colloc":    5000,
    "N_bord":      1000,
}


def sensibilite_pinn_architecture(params, f, tests, N_fd=100, u_ref=None, epoch_list=None, N_runs=10):
    """
    Compare plusieurs architectures PINN sur le même problème.
 
    Paramètres
    ----------
    params      : objet contenant X_min, X_max, Y_min, Y_max
    f           : forcing / paramètres du problème
    tests       : liste de tuples (label, N_layers, hidden_size)
                  ex: [('4x64', 4, 64), ('6x128', 6, 128)]
    N_fd        : résolution de la grille d'évaluation
    u_ref       : callable optionnel — solution analytique u_ref(XX, YY, params, xp)
    epoch_list  : liste d'époques cumulées, ex: [100, 250, 500, 1000]
    N_runs      : nombre de runs indépendants par architecture (pour mean ± std)
 
    Retourne
    --------
    results : dict  label -> dict de métriques (mean/std par checkpoint)
    mode    : str   'analytique' ou 'fd'
    """
    if epoch_list is None:
        epoch_list = [100, 250, 500, 1000]
 
    epoch_list = list(epoch_list)          # on s'assure d'avoir une liste Python
    max_epochs = epoch_list[-1]
 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
 
    # ── mode d'évaluation ────────────────────────────────────────────────────
    mode = "analytique" if u_ref is not None else "fd"
 
    # ── grille d'évaluation (calculée une seule fois) ────────────────────────
    XX = torch.linspace(params.X_min, params.X_max, N_fd).to(device)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd).to(device)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
    _, mask_PDE, _   = make_masks(XX_grid, YY_grid, params, xp=torch)
    mask_flat        = mask_PDE.flatten()
    XY               = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
 
    # ── solution de référence ────────────────────────────────────────────────
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()
    else:
        t0 = time.time()
        print(f"Calcul de la référence FD sur {N_fd}×{N_fd} points...")
        u_fd, _, _ = slv_fd.Solveur_Zermelo(N_fd - 1, f, params)
        u_ref_vals = torch.tensor(
            np.nan_to_num(u_fd).T.flatten(), dtype=torch.float32, device=device
        )
        print(f"Référence calculée en {time.time() - t0:.2f}s")
 
    # ── boucle sur les architectures ─────────────────────────────────────────
    results = {}
 
    for test in tests:
        label, N, H = test[0], test[1], test[2]
 
        hp = {
            "hidden_size": H,
            "n_layers"   : N,
            "N_colloc"   : 2000,
            "N_bord"     : 200,
            "lam"        : 1.0,
            "lr"         : 1e-3,
        }
 
        # accumulateurs : listes de listes  (N_runs × len(epoch_list))
        all_errors = []   # erreur L∞ à chaque checkpoint
        all_losses = []   # loss finale à chaque checkpoint
        all_times  = []   # durée de chaque segment d'entraînement
 
        for i in range(N_runs):
            print(f" Architecture : {label} ({N} couches, {H} neurones)|run {i + 1}/{N_runs}")
            run_errors = []
            run_losses = []
            run_times  = []
            trained_epochs = 0
 
            # un seul modèle par run, entraîné de façon incrémentale
            model = PINN(
                hidden_size=hp["hidden_size"],
                n_layers=hp["n_layers"]
            ).to(device)
 
            for epochs in epoch_list:
                n_new = epochs - trained_epochs   # epochs supplémentaires
 
                t0 = time.time()
                loss_history, _ = tr.train_SGD(
                    model, params, f,
                    N_colloc=hp["N_colloc"],
                    N_bord=hp["N_bord"],
                    n_epochs=n_new,
                    lam=hp["lam"],
                    lr=hp["lr"], 
                    norme = 'H1_2'
                )
                seg_time      = time.time() - t0
                trained_epochs = epochs
 
                loss_history = np.asarray(loss_history, dtype=float)
 
                with torch.no_grad():
                    u_pred  = model(XY).squeeze()
                    diff    = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
                    err_val = torch.max(diff[mask_flat]).item()
 
                run_errors.append(err_val)
                run_losses.append(float(loss_history[-1]))
                run_times.append(seg_time)
 
                print(f"    {epochs:5d} ep | loss = {run_losses[-1]:.4e} "
                      f"| err L∞ = {err_val:.4e} | Δt = {seg_time:.1f}s")
 
            all_errors.append(run_errors)
            all_losses.append(run_losses)
            all_times.append(run_times)
            del model
            torch.mps.empty_cache()
 
        # ── agrégation mean ± std ─────────────────────────────────────────────
        arr_err  = np.array(all_errors)   # (N_runs, len(epoch_list))
        arr_loss = np.array(all_losses)
        arr_time = np.array(all_times)
 
        err_mean  = arr_err.mean(axis=0)
        err_std   = arr_err.std(axis=0)
        loss_mean = arr_loss.mean(axis=0)
        loss_std  = arr_loss.std(axis=0)
 
        cumtimes      = np.cumsum(arr_time, axis=1)   # (N_runs, n_checkpoints)
        cumtime_mean  = cumtimes.mean(axis=0)
        cumtime_std   = cumtimes.std(axis=0)
 
        results[label] = {
            "err_mean"    : err_mean,
            "err_std"     : err_std,
            "loss_mean"   : loss_mean,
            "loss_std"    : loss_std,
            "cumtime_mean": cumtime_mean,
            "cumtime_std" : cumtime_std,
            "n_layers"    : N,
            "hidden_size" : H,
        }
 
    # ── visualisation ─────────────────────────────────────────────────────────
    epoch_arr = np.array(epoch_list)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors
 
    def _fill(ax, x, mean, std, color, alpha=0.20):
        """Bande ±1σ robuste en échelle log (clip à 1e-12 pour éviter ≤0)."""
        lo = np.clip(mean - std, 1e-12, None)
        hi = mean + std
        ax.fill_between(x, lo, hi, alpha=alpha, color=color)
 
    # — erreur L∞ vs époques cumulées ————————————————————————
    ax = axes[0]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(epoch_arr, res["err_mean"], marker='o', label=label, color=c)
        _fill(ax, epoch_arr, res["err_mean"], res["err_std"], c)
    ax.set_xlabel("Époques cumulées")
    ax.set_ylabel("Erreur $L^\\infty$ vs référence")
    ax.set_title("Erreur par architecture")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
 
    # — loss vs époques cumulées ——————————————————————————————
    ax = axes[1]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(epoch_arr, res["loss_mean"], marker='s', label=label, color=c)
        _fill(ax, epoch_arr, res["loss_mean"], res["loss_std"], c)
    ax.set_xlabel("Époques cumulées")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence de la loss")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)

 
    # — erreur L∞ vs temps cumulé —————————————————————————————
    ax = axes[2]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(res["cumtime_mean"], res["err_mean"], marker='D', label=label, color=c)
        _fill(ax, res["cumtime_mean"], res["err_mean"], res["err_std"], c)
    ax.set_xlabel("Temps cumulé (s)")
    ax.set_ylabel("Erreur $L^\\infty$ vs référence")
    ax.set_title("Erreur vs temps de calcul")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
 
    plt.suptitle(
        f"Sensibilité à l'architecture PINN — référence : {mode} "
        f"— mean ± 1σ sur {N_runs} runs",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"Architecture_{mode}.png", dpi=300)
    plt.close()
 
    return results, mode


def sensibilite_pinn_loss(
    params,
    f,
    loss_dict,
    N_fd=100,
    u_ref=None,
    epoch_list=None,
    n_layers=6,
    hidden_size=32,
    N_colloc=2000,
    N_bord=2000,
    N_runs=5,
):
    """
    Compare plusieurs fonctions de perte PINN sur le même problème,
    avec moyennes et écarts-types sur plusieurs runs.

    Paramètres
    ----------
    loss_dict   : dict {label: norme}
                  ex: {'L2': 'l2', 'H1/2': 'H1_2', 'H3/2': 'H3_2'}
    epoch_list  : liste d'époques cumulées, ex: [100, 250, 500, 1000]
    n_layers    : nombre de couches
    hidden_size : largeur du réseau
    N_colloc    : nombre de points de collocation
    N_bord      : nombre de points de bord
    N_runs      : nombre de runs indépendants
    u_ref       : callable optionnel (solution analytique)

    Retourne
    --------
    results : dict
    mode    : str
    """
    if epoch_list is None:
        epoch_list = [100, 250, 500, 1000]

    epoch_list = list(epoch_list)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ── mode d'évaluation ─────────────────────────────────────────────
    mode = "analytique" if u_ref is not None else "fd"

    # ── grille d'évaluation (calculée une seule fois) ────────────────
    XX = torch.linspace(params.X_min, params.X_max, N_fd, device=device)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd, device=device)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing="ij")
    _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp=torch)
    mask_flat = mask_PDE.flatten()
    XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)

    # ── solution de référence ─────────────────────────────────────────
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()
    else:
        t0 = time.time()
        print(f"Calcul de la référence FD sur {N_fd}×{N_fd} points...")
        u_fd, _, _ = slv_fd.Solveur_Zermelo(N_fd - 1, f, params)
        u_ref_vals = torch.tensor(
            np.nan_to_num(u_fd).T.flatten(), dtype=torch.float32, device=device
        )
        print(f"Référence calculée en {time.time() - t0:.2f}s")

    # ── boucle sur les fonctions de perte ─────────────────────────────
    results = {}

    for label, norme in loss_dict.items():
        print(f"\n── Norme : {label} ──")

        all_errors = []
        all_losses = []
        all_times = []

        for run in range(N_runs):
            print(f"  run {run + 1}/{N_runs}")

            model = PINN(n_layers=n_layers, hidden_size=hidden_size).to(device)

            run_errors = []
            run_losses = []
            run_times = []

            trained_epochs = 0

            for epochs in epoch_list:
                n_epochs_step = epochs - trained_epochs
                t_start = time.time()

                loss_history, _ = tr.train_SGD(
                    model,
                    params,
                    f,
                    N_colloc=N_colloc,
                    N_bord=N_bord,
                    n_epochs=n_epochs_step,
                    norme=norme,
                )

                seg_time = time.time() - t_start
                trained_epochs = epochs

                loss_history = np.asarray(loss_history, dtype=float)

                model.eval()
                with torch.no_grad():
                    u_pred = model(XY).squeeze()
                    diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
                    err = torch.max(diff[mask_flat]).item()

                run_errors.append(err)
                run_losses.append(float(loss_history[-1]))
                run_times.append(seg_time)

                print(
                    f"    {epochs:5d} ep | loss = {run_losses[-1]:.4e} "
                    f"| err L∞ = {run_errors[-1]:.4e} | Δt = {seg_time:.1f}s"
                )

            all_errors.append(run_errors)
            all_losses.append(run_losses)
            all_times.append(run_times)

            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        # ── agrégation ────────────────────────────────────────────────
        arr_err = np.array(all_errors)     # (N_runs, n_checkpoints)
        arr_loss = np.array(all_losses)
        arr_time = np.array(all_times)

        err_mean = arr_err.mean(axis=0)
        err_std = arr_err.std(axis=0)

        loss_mean = arr_loss.mean(axis=0)
        loss_std = arr_loss.std(axis=0)

        cumtimes = np.cumsum(arr_time, axis=1)
        cumtime_mean = cumtimes.mean(axis=0)
        cumtime_std = cumtimes.std(axis=0)

        results[label] = {
            "errors_mean": err_mean,
            "errors_std": err_std,
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "cumtime_mean": cumtime_mean,
            "cumtime_std": cumtime_std,
            "norme": norme,
        }

    # ── visualisation ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors
    epoch_arr = np.array(epoch_list)

    def _fill(ax, x, mean, std, color, alpha=0.20):
        lo = np.clip(mean - std, 1e-12, None)
        hi = mean + std
        ax.fill_between(x, lo, hi, alpha=alpha, color=color)

    # erreur L∞ vs époques
    ax = axes[0]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(epoch_arr, res["errors_mean"], marker="o", label=label, color=c)
        _fill(ax, epoch_arr, res["errors_mean"], res["errors_std"], c)
    ax.set_xlabel("Époques cumulées")
    ax.set_ylabel("Erreur $L^\\infty$ vs référence")
    ax.set_title("Erreur par fonction de perte")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # loss vs époques
    ax = axes[1]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(epoch_arr, res["loss_mean"], marker="s", label=label, color=c)
        _fill(ax, epoch_arr, res["loss_mean"], res["loss_std"], c)
    ax.set_xlabel("Époques cumulées")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence de la loss")
    ax.set_ylim(10**(-7), 0.1)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # erreur vs temps cumulé
    ax = axes[2]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(
            res["cumtime_mean"],
            res["errors_mean"],
            marker="D",
            label=label,
            color=c,
        )
        _fill(ax, res["cumtime_mean"], res["errors_mean"], res["errors_std"], c)
    ax.set_xlabel("Temps cumulé (s)")
    ax.set_ylabel("Erreur $L^\\infty$ vs référence")
    ax.set_title("Erreur vs temps de calcul")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.suptitle(
        f"Sensibilité à la fonction de perte — architecture {n_layers}×{hidden_size} "
        f"— référence : {mode} — mean ± 1σ sur {N_runs} runs",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"Pertes_{mode}.png", dpi=300)
    plt.close()

    return results, mode


def sensibilite_pinn_lcol(params, f, tests, N_fd=100, u_ref = None, N_runs = 10, epoch_list = None): 
    """
    Compare plusieurs architectures PINN sur le même problème.
 
    Paramètres
    ----------
    params      : objet contenant X_min, X_max, Y_min, Y_max
    f           : forcing / paramètres du problème
    tests       : liste de tuples (label, hidden_size, N_colloc)
    N_fd        : résolution de la grille d'évaluation
    u_ref       : callable optionnel — solution analytique u_ref(XX, YY, params, xp)
    epoch_list  : liste d'époques cumulées, ex: [100, 250, 500, 1000]
    N_runs      : nombre de runs indépendants par architecture (pour mean ± std)
 
    Retourne
    --------
    results : dict  label -> dict de métriques (mean/std par checkpoint)
    mode    : str   'analytique' ou 'fd'
    """
    if epoch_list is None:
        epoch_list = [100, 250, 500, 1000]
 
    epoch_list = list(epoch_list)          # on s'assure d'avoir une liste Python
    max_epochs = epoch_list[-1]
 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
 
    # ── mode d'évaluation ────────────────────────────────────────────────────
    mode = "analytique" if u_ref is not None else "fd"
 
    # ── grille d'évaluation (calculée une seule fois) ────────────────────────
    XX = torch.linspace(params.X_min, params.X_max, N_fd).to(device)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd).to(device)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
    _, mask_PDE, _   = make_masks(XX_grid, YY_grid, params, xp=torch)
    mask_flat        = mask_PDE.flatten()
    XY               = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
 
    # ── solution de référence ────────────────────────────────────────────────
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()
    else:
        t0 = time.time()
        print(f"Calcul de la référence FD sur {N_fd}×{N_fd} points...")
        u_fd, _, _ = slv_fd.Solveur_Zermelo(N_fd - 1, f, params)
        u_ref_vals = torch.tensor(
            np.nan_to_num(u_fd).T.flatten(), dtype=torch.float32, device=device
        )
        print(f"Référence calculée en {time.time() - t0:.2f}s")
 
    # ── boucle sur les architectures ─────────────────────────────────────────
    results = {}
    last_model_state = {}
    for test in tests:
        label, H, N = test[0], test[1], test[2]
 
        hp = {
            "hidden_size": H,
            "n_layers"   : 6,
            "N_colloc"   : N,
            "N_bord"     : N,
            "lam"        : 1.0,
            "lr"         : 1e-3,
        }
 
        # accumulateurs : listes de listes  (N_runs × len(epoch_list))
        all_errors = []   # erreur L∞ à chaque checkpoint
        all_losses = []   # loss finale à chaque checkpoint
        all_times  = []   # durée de chaque segment d'entraînement
 
        for i in range(N_runs):
            print(f" Architecture : {label} ({H} neurones, {N} points de colloc)|run {i + 1}/{N_runs}")
            run_errors = []
            run_losses = []
            run_times  = []
            trained_epochs = 0
 
            # un seul modèle par run, entraîné de façon incrémentale
            model = PINN(
                hidden_size=hp["hidden_size"],
                n_layers=hp["n_layers"]
            ).to(device)
 
            for epochs in epoch_list:
                n_new = epochs - trained_epochs   # epochs supplémentaires
 
                t0 = time.time()
                loss_history, _ = tr.train_SGD(
                    model, params, f,
                    N_colloc=hp["N_colloc"],
                    N_bord=hp["N_bord"],
                    n_epochs=n_new,
                    lam=hp["lam"],
                    lr=hp["lr"], 
                    norme = 'H1_2'
                )
                seg_time      = time.time() - t0
                trained_epochs = epochs
 
                loss_history = np.asarray(loss_history, dtype=float)
 
                with torch.no_grad():
                    u_pred  = model(XY).squeeze()
                    diff    = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
                    err_val = torch.max(diff[mask_flat]).item()
 
                run_errors.append(err_val)
                run_losses.append(float(loss_history[-1]))
                run_times.append(seg_time)
 
                print(f"    {epochs:5d} ep | loss = {run_losses[-1]:.4e} "
                      f"| err L∞ = {err_val:.4e} | Δt = {seg_time:.1f}s")
 
            all_errors.append(run_errors)
            all_losses.append(run_losses)
            all_times.append(run_times)
            if i == N_runs - 1:  # on garde uniquement le dernier run
                last_model_state[label] = {k: v.clone() for k, v in model.state_dict().items()}
            del model
            torch.mps.empty_cache()
 
        # ── agrégation mean ± std ─────────────────────────────────────────────
        arr_err  = np.array(all_errors)   # (N_runs, len(epoch_list))
        arr_loss = np.array(all_losses)
        arr_time = np.array(all_times)
 
        err_mean  = arr_err.mean(axis=0)
        err_std   = arr_err.std(axis=0)
        loss_mean = arr_loss.mean(axis=0)
        loss_std  = arr_loss.std(axis=0)
 
        cumtimes      = np.cumsum(arr_time, axis=1)   # (N_runs, n_checkpoints)
        cumtime_mean  = cumtimes.mean(axis=0)
        cumtime_std   = cumtimes.std(axis=0)
 
        results[label] = {
            "err_mean"    : err_mean,
            "err_std"     : err_std,
            "loss_mean"   : loss_mean,
            "loss_std"    : loss_std,
            "cumtime_mean": cumtime_mean,
            "cumtime_std" : cumtime_std,
            "n_colloc"    : N,
            "hidden_size" : H,
        }
 
    # ── visualisation ─────────────────────────────────────────────────────────
    epoch_arr = np.array(epoch_list)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors
 
    def _fill(ax, x, mean, std, color, alpha=0.20):
        """Bande ±1σ robuste en échelle log (clip à 1e-12 pour éviter ≤0)."""
        lo = np.clip(mean - std, 1e-12, None)
        hi = mean + std
        ax.fill_between(x, lo, hi, alpha=alpha, color=color)
 
    # — erreur L∞ vs époques cumulées ————————————————————————
    ax = axes[0]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(epoch_arr, res["err_mean"], marker='o', label=label, color=c)
        _fill(ax, epoch_arr, res["err_mean"], res["err_std"], c)
    ax.set_xlabel("Époques cumulées")
    ax.set_ylabel("Erreur $L^\\infty$ vs référence")
    ax.set_title("Erreur par architecture")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
 
    # — loss vs époques cumulées ——————————————————————————————
    ax = axes[1]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(epoch_arr, res["loss_mean"], marker='s', label=label, color=c)
        _fill(ax, epoch_arr, res["loss_mean"], res["loss_std"], c)
    ax.set_xlabel("Époques cumulées")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence de la loss")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)

 
    # — erreur L∞ vs temps cumulé —————————————————————————————
    ax = axes[2]
    for i, (label, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.loglog(res["cumtime_mean"], res["err_mean"], marker='D', label=label, color=c)
        _fill(ax, res["cumtime_mean"], res["err_mean"], res["err_std"], c)
    ax.set_xlabel("Temps cumulé (s)")
    ax.set_ylabel("Erreur $L^\\infty$ vs référence")
    ax.set_title("Erreur vs temps de calcul")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
 
    plt.suptitle(
        f"Sensibilité au couple largeur-collocation — référence : {mode} "
        f"— mean ± 1σ sur {N_runs} runs",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"largeur_col_{mode}.png", dpi=300)
    plt.close()
    # ── visualisation de la solution approchée (dernier test, dernier run) ───
    last_label = tests[-1][0]
    last_H     = tests[-1][1]

    model_display = PINN(hidden_size=last_H, n_layers=6).to(device)
    model_display.load_state_dict(last_model_state[last_label])
    model_display.eval()

    with torch.no_grad():
        u_pred_flat = model_display(XY).squeeze().cpu().numpy()
    del model_display

    XX_np = XX_grid.cpu().numpy()   # (N_fd, N_fd), indexing='ij' → axe 0 = x, axe 1 = y
    YY_np = YY_grid.cpu().numpy()

    # Masque 2D pour mettre NaN hors domaine (affichage propre)
    if mode == "fd":
        # mask_flat provient du solveur FD (bool tensor, intérieur strict)
        mask_2d = mask_flat.cpu().numpy().reshape(N_fd, N_fd)
    else:
        mask_2d = mask_PDE.cpu().numpy()

    u_pred_2d = u_pred_flat.reshape(N_fd, N_fd)
    u_ref_2d  = u_ref_vals.cpu().numpy().reshape(N_fd, N_fd)

    # On met NaN hors domaine pour que pcolormesh ne les colore pas
    u_pred_2d_plot = np.where(mask_2d, u_pred_2d, np.nan)
    u_ref_2d_plot  = np.where(mask_2d, u_ref_2d,  np.nan)
    err_2d_plot    = np.where(mask_2d, np.abs(u_pred_2d - u_ref_2d), np.nan)

    vmin = np.nanmin(u_ref_2d_plot)
    vmax = np.nanmax(u_ref_2d_plot)

    fig_sol, axes_sol = plt.subplots(1, 3, figsize=(15, 4.5))

    # — PINN ——————————————————————————————————————————————
    ax = axes_sol[0]
    im = ax.pcolormesh(XX_np, YY_np, u_pred_2d_plot,
                       cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"PINN — {last_label}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # — Référence ————————————————————————————————————————
    ax = axes_sol[1]
    im = ax.pcolormesh(XX_np, YY_np, u_ref_2d_plot,
                       cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Référence ({mode})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # — Erreur ————————————————————————————————————————————
    ax = axes_sol[2]
    im = ax.pcolormesh(XX_np, YY_np, err_2d_plot,
                       cmap="Reds", shading="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Erreur |PINN − ref| — {last_label}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.suptitle(
        f"Solution approchée — {last_label} — dernier run, {epoch_list[-1]} époques",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"solution_{last_label}.png", dpi=300)
    plt.close()
 
    return results, mode

def comparaison_pinn_fd_losses(
    params,
    f,
    N_list_fd,
    epoch_list,
    loss_dict,
    u_ref=None,
    N_runs=5,
    N_eval=100,
    pinn_kwargs=None,
    train_base_kwargs=None,
):
    """
    Compare plusieurs PINNs (différentes losses de bord) à une méthode FD
    en termes d'erreur L∞ vs temps de calcul.

    Paramètres
    ----------
    params : Params
        Paramètres du problème.
    f : callable
        Forcing / coefficient du problème.
    N_list_fd : list[int]
        Liste des tailles de grille pour la méthode FD.
    epoch_list : list[int]
        Liste des checkpoints cumulés pour le PINN.
    loss_dict : dict
        Dictionnaire {label_affiche: norme_train}
        ex: {"H1/2": "H1_2", "H3/2": "H3_2"}
    u_ref : callable or None
        Solution analytique de référence si disponible.
        Si None, la référence est calculée par FD sur une grille fine N_eval x N_eval.
    N_runs : int
        Nombre de runs indépendants par loss PINN.
    N_eval : int
        Taille de la grille d'évaluation du PINN, et taille de la référence FD si u_ref=None.
    pinn_kwargs : dict or None
        Arguments pour PINN(...), ex {"n_layers": 6, "hidden_size": 32}
    train_base_kwargs : dict or None
        Arguments communs pour tr.train(...), hors "norme".
        ex {"N_colloc": 500, "N_bord": 500}

    Retourne
    --------
    results : dict
        Résultats FD + PINN agrégés.
    mode : str
        "analytique" ou "fd"
    """
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    if pinn_kwargs is None:
        pinn_kwargs = {}

    if train_base_kwargs is None:
        train_base_kwargs = {}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mode = "analytique" if u_ref is not None else "fd"

    # ------------------------------------------------------------
    # 1. Partie FD
    # ------------------------------------------------------------
    print("Calcul des résultats FD...")
    if mode == "analytique":
        _, times_fd, err_fd = convergence_fd(params, N_list_fd, f, u_exacte=u_ref)
    else:
        # on utilise une référence FD fine via la fonction convergence_fd
        # ce qui suppose que ta convergence_fd gère déjà u_exacte=None
        # si ce n'est pas le cas, il faudra l'étendre
        _, times_fd, err_fd = convergence_fd(params, N_list_fd, f, u_exacte=None)

    times_fd = np.array(times_fd, dtype=float)
    err_fd = np.array(err_fd, dtype=float)

    # ------------------------------------------------------------
    # 2. Grille d'évaluation du PINN
    # ------------------------------------------------------------
    XX = torch.linspace(params.X_min, params.X_max, N_eval, device=device)
    YY = torch.linspace(params.Y_min, params.Y_max, N_eval, device=device)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing="ij")
    _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp=torch)
    mask_flat = mask_PDE.flatten()

    XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)

    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()
    else:
        print(f"Calcul de la référence FD fine sur une grille {N_eval}x{N_eval}...")
        u_fd_ref, _, _ = slv_fd.Solveur_Zermelo(N_eval - 1, f, params)
        u_ref_vals = torch.tensor(
            np.nan_to_num(u_fd_ref).T.flatten(),
            dtype=torch.float32,
            device=device
        )

    # ------------------------------------------------------------
    # 3. Partie PINN : une boucle par loss
    # ------------------------------------------------------------
    results_pinn = {}

    for loss_label, norme in loss_dict.items():
        print(f"\n=== PINN | loss = {loss_label} ({norme}) ===")

        all_errors = []
        all_times = []

        for run in range(N_runs):
            print(f"  run {run + 1}/{N_runs}")

            model = PINN(**pinn_kwargs).to(device)

            run_errors = []
            run_times = []
            trained_epochs = 0

            for epochs in epoch_list:
                n_new = epochs - trained_epochs

                _, tim = tr.train_SGD(
                    model,
                    params,
                    f,
                    n_epochs=n_new,
                    norme=norme,
                    **train_base_kwargs
                )
                seg_time = tim
                trained_epochs = epochs

                with torch.no_grad():
                    u_pred = model(XY).squeeze()

                diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
                err = torch.max(diff[mask_flat]).item()

                run_errors.append(err)
                run_times.append(seg_time)

                print(
                    f"    {epochs:5d} ep | err = {err:.4e} | Δt = {seg_time:.2f}s"
                )

            all_errors.append(run_errors)
            all_times.append(run_times)

            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        arr_err = np.array(all_errors, dtype=float)
        arr_time = np.array(all_times, dtype=float)

        err_mean = arr_err.mean(axis=0)
        err_std = arr_err.std(axis=0)

        cumtime = np.cumsum(arr_time, axis=1)
        cumtime_mean = cumtime.mean(axis=0)
        cumtime_std = cumtime.std(axis=0)

        results_pinn[loss_label] = {
            "norme": norme,
            "err_mean": err_mean,
            "err_std": err_std,
            "cumtime_mean": cumtime_mean,
            "cumtime_std": cumtime_std,
        }

    # ------------------------------------------------------------
    # 4. Visualisation
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    colors = plt.cm.tab10.colors

    for i, (loss_label, res) in enumerate(results_pinn.items()):
        c = colors[i % len(colors)]

        plt.loglog(
            res["cumtime_mean"],
            res["err_mean"],
            marker="o",
            label=f"PINN {loss_label}",
            color=c
        )
        plt.fill_between(
            res["cumtime_mean"],
            np.clip(res["err_mean"] - res["err_std"], 1e-12, None),
            res["err_mean"] + res["err_std"],
            alpha=0.2,
            color=c
        )

    plt.loglog(
        times_fd,
        err_fd,
        marker="s",
        linestyle="--",
        label="Différences finies",
    )

    plt.xlabel("Temps de calcul (s)")
    plt.ylabel("Erreur $L^\\infty$")
    plt.title(f"Comparaison PINN vs différences finies ({mode})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    filename = f"Comparaison_PINN_FD_{mode}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    results = {
        "fd": {
            "N_list": np.array(N_list_fd),
            "times": times_fd,
            "errors": err_fd,
        },
        "pinn": results_pinn,
        "mode": mode,
        "filename": filename,
        "epoch_list": np.array(epoch_list),
        "pinn_kwargs": pinn_kwargs,
        "train_base_kwargs": train_base_kwargs,
    }

    return results, mode




def convergence_policy_iteration(
    params,
    f,
    vc_field,
    N_fd=1000,
    u_ref=None,
    n_policy_iter=10,
    lr=1e-3,
    eta0=1.0,
    eta_schedule=None,
    min_epochs=200,
    max_epochs=3000,
    tol_policy=1e-4,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    N_colloc=1000,
):
    """
    Analyse de convergence de la policy iteration.

    Retourne
    --------
    errors : list[float]
        erreur vs référence à chaque itération externe
    times : list[float]
        temps cumulé
    policy_errors : list[float]
        ||u_{k+1} - u_k||_{H2,h} sur la grille d'évaluation
    loss_histories : list[np.ndarray]
        historique de loss pour chaque sous-problème de policy iteration
    mode : str
        'analytique' ou 'fd'
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if u_ref is not None:
        mode = "analytique"
    else:
        mode = "fd"

    if eta_schedule is None:
        eta_schedule = [0.5 ** k for k in range(n_policy_iter)]

    if len(eta_schedule) < n_policy_iter:
        raise ValueError("eta_schedule est trop courte pour n_policy_iter")

    # ===============================
    # GRILLE D'ÉVALUATION
    # ===============================
    XX = torch.linspace(params.X_min, params.X_max, N_fd, device=device)
    YY = torch.linspace(params.Y_min, params.Y_max, N_fd, device=device)
    XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing="ij")

    _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp=torch)
    XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)

    # ===============================
    # SOLUTION DE RÉFÉRENCE
    # ===============================
    if mode == "analytique":
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()

        # dérivées de référence éventuelles
        XY_req = XY.clone().detach().requires_grad_(True)
        u_ref_xy = u_ref(
            XY_req[:, 0:1], XY_req[:, 1:2], params, xp=torch
        )
        grads = torch.autograd.grad(u_ref_xy.sum(), XY_req, create_graph=True)[0]
        ux_ref = grads[:, 0:1]
        uy_ref = grads[:, 1:2]
        uxx_ref = torch.autograd.grad(ux_ref.sum(), XY_req, create_graph=True)[0][:, 0:1]
        uyy_ref = torch.autograd.grad(uy_ref.sum(), XY_req, create_graph=True)[0][:, 1:2]

    else:
        print(f"Calcul FD référence ({N_fd}x{N_fd})...")
        u_fd, _, _ = slv_fd.Solveur_Zermelo(N_fd - 1, f, params)
        u_ref_vals = torch.tensor(
            np.nan_to_num(u_fd).T.flatten(),
            dtype=torch.float32,
            device=device,
        )
        ux_ref = uy_ref = uxx_ref = uyy_ref = None

    # ===============================
    # INITIALISATION
    # ===============================
    model = PINN().to(device)

    errors = []
    times = []
    policy_errors = []
    loss_histories = []

    t_total = 0.0

    # ===============================
    # BOUCLE POLICY ITERATION
    # ===============================
    for k in range(n_policy_iter):
        eta_k = eta_schedule[k]
        print(f"\n=== Policy iteration {k} | eta_k={eta_k:.3e} ===")

        t0 = time.time()

        # sauvegarde de u_k
        model_old = copy.deepcopy(model)
        model_old.eval()

        # résolution du sous-problème linéarisé gelé
        model, history_inner = tr.train_policy_step(
            model=model,
            model_policy=model_old,
            params=params,
            vc_field=vc_field,
            f=f,
            eta_k=eta_k,
            eta0=eta0,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            lr=lr,
            boundary_mode=boundary_mode,
            gamma_boundary=gamma_boundary,
            verbose=False,
            N_colloc=N_colloc,
            device=device,
        )

        # stockage de l'historique de loss du sous-problème courant
        loss_histories.append(np.asarray(history_inner["loss"], dtype=float))

        t1 = time.time()
        t_total += (t1 - t0)
        times.append(t_total)

        # ===============================
        # ERREUR vs référence
        # ===============================
        model.eval()
        with torch.no_grad():
            u_pred = model(XY).squeeze()

        diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
        err = torch.max(diff[mask_PDE.flatten()]).item()
        errors.append(err)

        # ===============================
        # ERREUR POLICY en H² discrète
        # ===============================
        err_pol = tr.policy_iteration_error(model_old, model, XY)
        policy_errors.append(err_pol)

        print(f"error = {err:.3e} | policy_error(H2) = {err_pol:.3e}")

        if err_pol < tol_policy:
            print("Convergence de la policy iteration atteinte.")
            break

    return errors, times, policy_errors, loss_histories, mode


def plot_convergence_policy(errors, times, policy_errors):
    plt.figure(figsize=(12, 4))

    # erreur vs itération
    plt.subplot(1, 3, 1)
    plt.plot(errors, marker="o")
    plt.yscale("log")
    plt.title("Erreur vs policy iteration")
    plt.xlabel("k")

    # temps vs erreur
    plt.subplot(1, 3, 2)
    plt.plot(times, errors, marker="o")
    plt.yscale("log")
    plt.title("Erreur vs temps")
    plt.xlabel("temps (s)")

    # convergence interne
    plt.subplot(1, 3, 3)
    plt.plot(policy_errors, marker="o")
    plt.yscale("log")
    plt.title("||u_k - u_{k-1}||")
    plt.xlabel("k")

    plt.tight_layout()
    plt.show()

def run_policy_config(
    params,
    f,
    vc_field,
    u_ref=None,
    N_fd=200,
    n_policy_iter=5,
    lr=1e-3,
    eta0=1.0,
    eta_schedule=None,
    min_epochs=500,
    max_epochs=3000,
    tol_policy=1e-4,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    N_colloc=1000,
    device=None,
):
    """
    Lance un run de policy iteration et retourne un résumé exploitable
    pour les analyses et les tracés.

    Hypothèse :
    convergence_policy_iteration retourne désormais :
        errors, times, policy_errors, loss_histories, mode
    où loss_histories est une liste de listes/arrays,
    une par itération de policy iteration.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # NB : model_class n'était pas utilisé dans ta version initiale.
    # Si convergence_policy_iteration construit lui-même le modèle, il faut
    # lui transmettre model_class. Sinon, enlève cet argument de la signature.
    errors, times, policy_errors, loss_histories, mode = convergence_policy_iteration(
        params=params,
        f=f,
        vc_field=vc_field,
        N_fd=N_fd,
        u_ref=u_ref,
        n_policy_iter=n_policy_iter,
        lr=lr,
        eta0=eta0,
        eta_schedule=eta_schedule,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        tol_policy=tol_policy,
        boundary_mode=boundary_mode,
        gamma_boundary=gamma_boundary,
        N_colloc=N_colloc,
    )

    best_idx = int(np.argmin(errors))

    return {
        "errors": np.asarray(errors, dtype=float),
        "times": np.asarray(times, dtype=float),
        "policy_errors": np.asarray(policy_errors, dtype=float),
        "loss_histories": [np.asarray(loss, dtype=float) for loss in loss_histories],
        "best_error": float(errors[best_idx]),
        "best_time": float(times[best_idx]),
        "final_error": float(errors[-1]),
        "mode": mode,
    }


def sensibilite_policy_architecture_collocation(
    params,
    f,
    vc_field,
    model_factory,
    configs,
    u_ref=None,
    N_fd=200,
    n_policy_iter=8,
    lr=1e-3,
    eta0=10.0,
    eta_schedule=None,
    min_epochs=200,
    max_epochs=3000,
    tol_policy=1e-4,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    n_runs=5,
):
    """
    Teste plusieurs couples (H, N_colloc) dans l’esprit de la Figure 2 du papier.

    configs : liste de dicts, par ex.
        [
            {"name": "H80_N1000", "model_kwargs": {"hidden_dim": 80, "depth": 4}, "N_colloc": 1000},
            {"name": "H100_N1000", "model_kwargs": {"hidden_dim": 100, "depth": 4}, "N_colloc": 1000},
            {"name": "H80_N3000", "model_kwargs": {"hidden_dim": 80, "depth": 4}, "N_colloc": 3000},
        ]
    """
    if eta_schedule is None:
        eta_schedule = [0.5**k for k in range(n_policy_iter)]  # version pratique

    results = {}

    for cfg in configs:
        name = cfg["name"]
        model_kwargs = cfg["model_kwargs"]
        N_colloc = cfg["N_colloc"]

        runs = []
        for _ in range(n_runs):
            run = run_policy_config(
                params=params,
                f=f,
                vc_field=vc_field,
                u_ref=u_ref,
                N_fd=N_fd,
                n_policy_iter=n_policy_iter,
                lr=lr,
                eta0=eta0,
                eta_schedule=eta_schedule,
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                tol_policy=tol_policy,
                boundary_mode=boundary_mode,
                gamma_boundary=gamma_boundary,
                N_colloc=N_colloc,
            )
            runs.append(run)

        results[name] = runs

    return results

def sensibilite_policy_eta0(
    params,
    f,
    vc_field,
    model_class,
    eta0_list,
    u_ref=None,
    N_fd=200,
    n_policy_iter=8,
    lr=1e-3,
    eta_decay="2^-k",
    min_epochs=200,
    max_epochs=3000,
    tol_policy=1e-4,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    N_colloc=1000,
    n_runs=5,
):
    """
    Étudie l’impact de eta0 à architecture et collocation fixées.
    """
    results = {}

    for eta0 in eta0_list:
        if eta_decay == "2^-k":
            eta_schedule = [eta0] + [2**(-k) for k in range(1, n_policy_iter)]
        elif eta_decay == "4^-k":
            eta_schedule = [eta0] + [4**(-k) for k in range(1, n_policy_iter)]
        elif eta_decay == "1/k":
            eta_schedule = [eta0] + [1.0 / k for k in range(1, n_policy_iter)]
        else:
            raise ValueError("eta_decay inconnu")

        runs = []
        for _ in range(n_runs):
            run = run_policy_config(
                params=params,
                f=f,
                vc_field=vc_field,
                u_ref=u_ref,
                N_fd=N_fd,
                n_policy_iter=n_policy_iter,
                lr=lr,
                eta0=eta0,
                eta_schedule=eta_schedule,
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                tol_policy=tol_policy,
                boundary_mode=boundary_mode,
                gamma_boundary=gamma_boundary,
                N_colloc=N_colloc,
            )
            runs.append(run)

        results[f"eta0={eta0}"] = runs

    return results

def sensibilite_policy_eta_schedule(
    params,
    f,
    vc_field,
    model_class,
    schedule_configs,
    u_ref=None,
    N_fd=200,
    n_policy_iter=8,
    lr=1e-3,
    min_epochs=200,
    max_epochs=3000,
    tol_policy=1e-4,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    N_colloc=1000,
    n_runs=5,
):
    """
    schedule_configs : liste de dicts, par ex.
    [
        {"name": "eta0=20_2^-k", "eta0": 20.0, "eta_schedule": [20.0] + [2**(-k) for k in range(1, 8)]},
        {"name": "eta0=20_4^-k", "eta0": 20.0, "eta_schedule": [20.0] + [4**(-k) for k in range(1, 8)]},
        {"name": "eta0=40_4^-k", "eta0": 40.0, "eta_schedule": [40.0] + [4**(-k) for k in range(1, 8)]},
    ]
    """
    results = {}

    for cfg in schedule_configs:
        runs = []
        for _ in range(n_runs):
            run = run_policy_config(
                params=params,
                f=f,
                vc_field=vc_field,
                u_ref=u_ref,
                N_fd=N_fd,
                n_policy_iter=n_policy_iter,
                lr=lr,
                eta0=cfg["eta0"],
                eta_schedule=cfg["eta_schedule"],
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                tol_policy=tol_policy,
                boundary_mode=boundary_mode,
                gamma_boundary=gamma_boundary,
                N_colloc=N_colloc,
            )
            runs.append(run)

        results[cfg["name"]] = runs

    return results


def summarize_policy_runs(results):
    """
    Résume mean ± std sur :
    - best_error
    - best_time
    - final_error
    - best_loss
    - final_loss
    """

    summary = {}

    for name, runs in results.items():
        best_errors = np.array([r["best_error"] for r in runs], dtype=float)
        best_times  = np.array([r["best_time"] for r in runs], dtype=float)
        final_errors = np.array([r["final_error"] for r in runs], dtype=float)

        # ---- LOSS EXTRACTION ----
        best_losses = []
        final_losses = []

        for r in runs:
            if "loss_histories" not in r or len(r["loss_histories"]) == 0:
                continue

            # concat toutes les policy iterations
            full_loss = np.concatenate(r["loss_histories"])

            best_losses.append(np.min(full_loss))
            final_losses.append(full_loss[-1])

        best_losses = np.array(best_losses, dtype=float)
        final_losses = np.array(final_losses, dtype=float)

        summary[name] = {
            # erreurs
            "best_error_mean": best_errors.mean(),
            "best_error_std": best_errors.std(),
            "final_error_mean": final_errors.mean(),
            "final_error_std": final_errors.std(),

            # temps
            "best_time_mean": best_times.mean(),
            "best_time_std": best_times.std(),

            # loss
            "best_loss_mean": best_losses.mean() if len(best_losses) > 0 else np.nan,
            "best_loss_std": best_losses.std() if len(best_losses) > 0 else np.nan,
            "final_loss_mean": final_losses.mean() if len(final_losses) > 0 else np.nan,
            "final_loss_std": final_losses.std() if len(final_losses) > 0 else np.nan,
        }

    return summary

def plot_policy_results(results, title="policy_iteration_sensitivity", results_dir="results"):
    """
    Sauvegarde trois figures dans results_dir :
    1. erreur moyenne ± std en fonction du temps cumulé
    2. loss history moyenne ± std en fonction des epochs cumulées
    3. résumé agrégé des métriques (barplot)

    Ne retourne pas les graphes, ne les affiche pas.
    """
    os.makedirs(results_dir, exist_ok=True)

    saved_paths = {}

    # ==========================================================
    # 1) ERREUR VS TEMPS
    # ==========================================================
    plt.figure(figsize=(7, 5))

    for name, runs in results.items():
        if len(runs) == 0:
            continue

        max_len = max(len(r["errors"]) for r in runs)

        errs = np.full((len(runs), max_len), np.nan)
        times = np.full((len(runs), max_len), np.nan)

        for i, r in enumerate(runs):
            errs[i, :len(r["errors"])] = np.asarray(r["errors"], dtype=float)
            times[i, :len(r["times"])] = np.asarray(r["times"], dtype=float)

        mean_err = np.nanmean(errs, axis=0)
        std_err = np.nanstd(errs, axis=0)
        mean_time = np.nanmean(times, axis=0)

        plt.plot(mean_time, mean_err, marker="o", label=name)
        plt.fill_between(
            mean_time,
            mean_err - std_err,
            mean_err + std_err,
            alpha=0.2
        )

    plt.yscale("log")
    plt.xlabel("Temps cumulé (s)")
    plt.ylabel("Erreur")
    plt.title(title.replace("_", " ") + " - error vs time")
    plt.legend()
    plt.grid(True)

    error_time_path = os.path.join(results_dir, f"{title}_error_vs_time.pdf")
    plt.savefig(error_time_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["error_vs_time"] = error_time_path

    # ==========================================================
    # 2) LOSS HISTORY
    # ==========================================================
    plt.figure(figsize=(7, 5))
    has_loss_data = False

    for name, runs in results.items():
        if len(runs) == 0:
            continue

        concatenated_losses = []

        for r in runs:
            loss_histories = r.get("loss_histories", [])

            if len(loss_histories) == 0:
                concatenated_losses.append(np.array([], dtype=float))
                continue

            valid_losses = [
                np.asarray(loss, dtype=float)
                for loss in loss_histories
                if len(loss) > 0
            ]

            if len(valid_losses) == 0:
                concatenated_losses.append(np.array([], dtype=float))
            else:
                concatenated_losses.append(np.concatenate(valid_losses))

        max_len = max(len(loss) for loss in concatenated_losses)
        if max_len == 0:
            continue

        has_loss_data = True
        loss_mat = np.full((len(concatenated_losses), max_len), np.nan)

        for i, loss in enumerate(concatenated_losses):
            if len(loss) > 0:
                loss_mat[i, :len(loss)] = loss

        mean_loss = np.nanmean(loss_mat, axis=0)
        std_loss = np.nanstd(loss_mat, axis=0)
        epochs = np.arange(1, max_len + 1)

        plt.plot(epochs, mean_loss, label=name)
        plt.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.2
        )

    if has_loss_data:
        plt.yscale("log")
        plt.xlabel("Epochs cumulées")
        plt.ylabel("Loss")
        plt.title(title.replace("_", " ") + " - loss history")
        plt.legend()
        plt.grid(True)

        loss_history_path = os.path.join(results_dir, f"{title}_loss_history.pdf")
        plt.savefig(loss_history_path, dpi=300, bbox_inches="tight")
        saved_paths["loss_history"] = loss_history_path

    plt.close()

    # ==========================================================
    # 3) ERROR SUMMARY (barplot séparé)
    # ==========================================================
    labels = []
    best_error_mean = []
    best_error_std = []
    final_error_mean = []
    final_error_std = []

    for name, runs in results.items():
        if len(runs) == 0:
            continue

        labels.append(name)

        best_errors = np.array([r["best_error"] for r in runs], dtype=float)
        final_errors = np.array([r["final_error"] for r in runs], dtype=float)

        best_error_mean.append(np.mean(best_errors))
        best_error_std.append(np.std(best_errors))
        final_error_mean.append(np.mean(final_errors))
        final_error_std.append(np.std(final_errors))

    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(
            x - width / 2,
            best_error_mean,
            width,
            yerr=best_error_std,
            capsize=4,
            label="best error"
        )
        plt.bar(
            x + width / 2,
            final_error_mean,
            width,
            yerr=final_error_std,
            capsize=4,
            label="final error"
        )

        plt.xticks(x, labels, rotation=20)
        plt.yscale("log")
        plt.ylabel("Erreur")
        plt.title(title.replace("_", " ") + " - error summary")
        plt.legend()
        plt.grid(True, axis="y")

        error_summary_path = os.path.join(results_dir, f"{title}_error_summary.pdf")
        plt.savefig(error_summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths["error_summary"] = error_summary_path

    # ==========================================================
    # 4) LOSS SUMMARY (barplot séparé)
    # ==========================================================
    labels_loss = []
    best_loss_mean = []
    best_loss_std = []
    final_loss_mean = []
    final_loss_std = []

    for name, runs in results.items():
        if len(runs) == 0:
            continue

        run_best_losses = []
        run_final_losses = []

        for r in runs:
            loss_histories = r.get("loss_histories", [])
            if len(loss_histories) == 0:
                continue

            valid_losses = [
                np.asarray(loss, dtype=float)
                for loss in loss_histories
                if len(loss) > 0
            ]
            if len(valid_losses) == 0:
                continue

            full_loss = np.concatenate(valid_losses)
            run_best_losses.append(np.min(full_loss))
            run_final_losses.append(full_loss[-1])

        if len(run_best_losses) == 0:
            continue

        labels_loss.append(name)

        run_best_losses = np.asarray(run_best_losses, dtype=float)
        run_final_losses = np.asarray(run_final_losses, dtype=float)

        best_loss_mean.append(np.mean(run_best_losses))
        best_loss_std.append(np.std(run_best_losses))
        final_loss_mean.append(np.mean(run_final_losses))
        final_loss_std.append(np.std(run_final_losses))

    if len(labels_loss) > 0:
        x = np.arange(len(labels_loss))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(
            x - width / 2,
            best_loss_mean,
            width,
            yerr=best_loss_std,
            capsize=4,
            label="best loss"
        )
        plt.bar(
            x + width / 2,
            final_loss_mean,
            width,
            yerr=final_loss_std,
            capsize=4,
            label="final loss"
        )

        plt.xticks(x, labels_loss, rotation=20)
        plt.yscale("log")
        plt.ylabel("Loss")
        plt.title(title.replace("_", " ") + " - loss summary")
        plt.legend()
        plt.grid(True, axis="y")

        loss_summary_path = os.path.join(results_dir, f"{title}_loss_summary.pdf")
        plt.savefig(loss_summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths["loss_summary"] = loss_summary_path

    return saved_paths