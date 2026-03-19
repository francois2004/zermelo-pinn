""" Module de tests numériques de convergence des méthodes numériques pour les deux solution"""
import numpy as np
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
        u_ref_vals = torch.tensor(np.nan_to_num(u_fd), dtype=torch.float32, device=device).flatten()
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
    "N_colloc":    5000,
    "N_bord":      1000,
}

def sensibilité_pinn(params,hyper_params, f, max_epochs  = 100, u_ref = None, N_fd = 1000): 
    """
    Étudie la sensibilité du PINN à différents hyperparamètres (ex: lr, lam, N_colloc, N_bord).
    Renvoie les erreurs de convergence pour chaque configuration d'hyperparamètres.
    """
    ## référence précalculée en une fois : 
    if u_ref is not None:
        mode = "analytique"
    else:
        mode = "fd"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ## grille d'évaluation de la solution (calculée une seule fois)
    if mode == "analytique":
        XX = torch.linspace(params.X_min, params.X_max, N_fd).to(device)
        YY = torch.linspace(params.Y_min, params.Y_max, N_fd).to(device)
        XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
        _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp=torch)
        XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()
        mask_valid = mask_PDE.flatten()

    else:
        t0 = time.time()
        print(f"Calcul référence FD sur grille {N_fd}×{N_fd}...")
        u_fd, XX_fd, YY_fd = slv_fd.Solveur_Zermelo(N_fd - 1, f, params)

        # ✅ Grille, masque et référence tous alignés sur les mêmes points
        _, mask_PDE, _ = make_masks(XX_fd, YY_fd, params)
        XY = torch.tensor(
            np.stack([XX_fd.ravel(), YY_fd.ravel()], axis=1),
           dtype=torch.float32
        ).to(device)
        u_ref_vals = torch.tensor(u_fd, dtype=torch.float32).flatten().to(device)
        mask_valid = torch.tensor(
        mask_PDE.flatten() & ~np.isnan(u_fd.flatten())
        )
        print(f"Référence FD prête en {time.time()-t0:.2f}s")


        #print(f"U_fd min/max          : {np.nanmin(u_fd):.3f} / {np.nanmax(u_fd):.3f}")
        #print(f"NaN dans U_fd         : {np.isnan(u_fd).sum()}")
        #print(f"Valeur bord intérieur : {u_fd[u_fd.shape[0]//2, u_fd.shape[1]//2]:.3f}")

    # Bord intérieur — doit valoir 0
    # Bord extérieur — doit valoir 1
    # Coin du domaine — doit valoir NaN
        #print(f"Coin (0,0)            : {u_fd[0,0]}")

    ## boucle Principale
    results = {}

    for hp_name, hp_values in hyper_params.items():
        print(f"\n {'-'*55}")
        print(f" Hyperparamètre : {hp_name}| valeurs : {hp_values}")

        results[hp_name] = {}

        for val in hp_values :
            hp = {**DEFAULTS, hp_name : val}
            print(f"-> {hp_name} = {val} (autres : Défaut)")

            model = PINN(hidden_size=hp["hidden_size"], n_layers= hp["n_layers"]).to(device)

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
                u_pred = model(XY).squeeze().to(device)
                #print(f"u_pred min/max : {u_pred[mask_valid].min():.3f} / {u_pred[mask_valid].max():.3f}")
                #print(f"u_ref  min/max : {u_ref_vals[mask_valid].min():.3f} / {u_ref_vals[mask_valid].max():.3f}")
             ## erreur L_inf par rapport à la solution de référence (analytique ou FD)
            diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
            err = torch.max(diff[mask_valid])

            results[hp_name][val] = {
                "loss" : loss_hist, 
                "time" : total_time,
                "error" : err,
            }

            print(f"loss_finale = {loss_hist[-1]}"
                  f"err L_infty = {err:.3e} " f"temps = {total_time :.1f}s")
    return results


def sensibilite_pinn_2(params :Params, hyper_params : dict , f, max_epochs = 100, u_ref = None, N_fd = 1000, N_runs = 10 ):
    """
    Méthode qui étudie l'effet sur les pinns des hyperparamètres, comparée a une solution benchmarkée

    Paramètres : 
    -params : Params
        contient les paramètres de l'equation
    -hyper_params : dict
        contient les hyperparamètres a tester sur le Pinn
    - f : Callable
        fonction vent dans l'equation
    - max_epochs : int
        nombre d'episodes d'optimisation
    -u_ref : Callable = None
        solution de reference a l'equation si disponible, None sinon
    -N_fd : int = 1000
        nombre de points de discretisation pour la différence finie(benchmark si pas de u_ref dispo)
    - N_runs : int = 10
        nombre d'entrainement par hyperparamètre pour estimation de l'impact

    Retourne : 

    """ 
    ## Calcul de la solution de référence
    if u_ref is not None:
        mode = "analytique"
    else:
        mode = "fd"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ## grille d'évaluation de la solution (calculée une seule fois)
    if mode == "analytique":
        XX = torch.linspace(params.X_min, params.X_max, N_fd).to(device)
        YY = torch.linspace(params.Y_min, params.Y_max, N_fd).to(device)
        XX_grid, YY_grid = torch.meshgrid(XX, YY, indexing='ij')
        _, mask_PDE, _ = make_masks(XX_grid, YY_grid, params, xp=torch)
        XY = torch.stack([XX_grid.flatten(), YY_grid.flatten()], dim=1)
        u_ref_vals = u_ref(XX_grid, YY_grid, params, xp=torch).flatten()
        mask_valid = mask_PDE.flatten()

    else:
        t0 = time.time()
        print(f"Calcul référence FD sur grille {N_fd}×{N_fd}...")
        u_fd, XX_fd, YY_fd = slv_fd.Solveur_Zermelo(N_fd - 1, f, params)

        # ✅ Grille, masque et référence tous alignés sur les mêmes points
        _, mask_PDE, _ = make_masks(XX_fd, YY_fd, params)
        XY = torch.tensor(
            np.stack([XX_fd.ravel(), YY_fd.ravel()], axis=1),
           dtype=torch.float32
        ).to(device)
        u_ref_vals = torch.tensor(u_fd, dtype=torch.float32).flatten().to(device)
        mask_valid = torch.tensor(mask_PDE.flatten() & ~np.isnan(u_fd.flatten()), device = device)
        print(f"Référence FD prête en {time.time()-t0:.2f}s")
    
    ## Boucle principale
    results = {}

    for hp_name, hp_values in hyper_params.items():
        print(f"\n{'-'*55}")
        print(f"Hyperparamètre : {hp_name} | valeurs : {hp_values}")

        results[hp_name] = {}

        for val in hp_values:
            hp = {**DEFAULTS, hp_name: val}
            print(f"-> {hp_name} = {val} (autres = défaut)")

            loss_histories = []
            final_losses = []
            times = []
            errors = []

            for i in range(N_runs):

                model = PINN(
                    hidden_size=hp["hidden_size"],
                    n_layers=hp["n_layers"]
                ).to(device)

                t0 = time.time()
                loss_history, _ = tr.train(
                    model, params, f,
                    N_colloc=hp["N_colloc"],
                    N_bord=hp["N_bord"],
                    n_epochs=max_epochs,
                    lam=hp["lam"],
                    lr=hp["lr"]
                )
                run_time = time.time() - t0

                loss_history = np.asarray(loss_history, dtype=float)
                loss_histories.append(loss_history)
                final_losses.append(float(loss_history[-1]))
                times.append(run_time)

                with torch.no_grad():
                    u_pred = model(XY).squeeze()
                    diff = torch.abs(u_pred.flatten() - u_ref_vals.flatten())
                    err_val = torch.max(diff[mask_valid]).item()

                errors.append(err_val)

            results[hp_name][val] = {
                "loss_histories": loss_histories,
                "final_losses": final_losses,
                "times": times,
                "errors": errors,
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": float(np.std(final_losses)),
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
            }

            print(
                f"loss finale = {np.mean(final_losses):.3e} ± {np.std(final_losses):.3e} | "
                f"err L_inf = {np.mean(errors):.3e} ± {np.std(errors):.3e} | "
                f"temps = {np.mean(times):.2f} ± {np.std(times):.2f}s"
            )

    return results

def plot_sensibilite_stats(results, show_loss_hist=True, logy_error=True, logy_loss=True):
    """
    Trace les résultats produits par sensibilite_pinn_2.

    Paramètres
    ----------
    results : dict
        Dictionnaire retourné par sensibilite_pinn_2.
    show_loss_hist : bool
        Si True, trace aussi la loss moyenne au cours des epochs.
    logy_error : bool
        Si True, met l'axe y de l'erreur en échelle logarithmique.
    logy_loss : bool
        Si True, met l'axe y des losses en échelle logarithmique.
    """

    for hp_name, hp_dict in results.items():
        if len(hp_dict) == 0:
            continue

        # -----------------------------
        # Extraction + tri des données
        # -----------------------------
        vals = []
        mean_errors = []
        std_errors = []
        mean_times = []
        std_times = []
        mean_losses = []
        std_losses = []
        all_loss_histories = []

        for val, d in hp_dict.items():
            vals.append(float(val))
            mean_errors.append(d["mean_error"])
            std_errors.append(d["std_error"])
            mean_times.append(d["mean_time"])
            std_times.append(d["std_time"])
            mean_losses.append(d["mean_final_loss"])
            std_losses.append(d["std_final_loss"])
            all_loss_histories.append(d["loss_histories"])

        vals = np.array(vals)
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)
        mean_times = np.array(mean_times)
        std_times = np.array(std_times)
        mean_losses = np.array(mean_losses)
        std_losses = np.array(std_losses)

        order = np.argsort(vals)
        vals = vals[order]
        mean_errors = mean_errors[order]
        std_errors = std_errors[order]
        mean_times = mean_times[order]
        std_times = std_times[order]
        mean_losses = mean_losses[order]
        std_losses = std_losses[order]
        all_loss_histories = [all_loss_histories[i] for i in order]

        # Si toutes les valeurs sont > 0, on peut envisager une échelle log en x
        use_logx = np.all(vals > 0)

        # -----------------------------
        # Figure principale : 3 panneaux
        # -----------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle(f"Sensibilité du PINN à {hp_name}", fontsize=14)

        # 1. Erreur
        ax = axes[0]
        ax.errorbar(vals, mean_errors, yerr=std_errors, fmt='o-', capsize=4)
        ax.set_xlabel(hp_name)
        ax.set_ylabel("Erreur $L_\\infty$")
        ax.set_title("Erreur moyenne ± écart-type")
        ax.grid(True)
        if logy_error:
            ax.set_yscale("log")
        if use_logx:
            ax.set_xscale("log")

        # 2. Temps
        ax = axes[1]
        ax.errorbar(vals, mean_times, yerr=std_times, fmt='s-', capsize=4)
        ax.set_xlabel(hp_name)
        ax.set_ylabel("Temps (s)")
        ax.set_title("Temps moyen ± écart-type")
        ax.grid(True)
        if use_logx:
            ax.set_xscale("log")

        # 3. Loss finale
        ax = axes[2]
        ax.errorbar(vals, mean_losses, yerr=std_losses, fmt='^-', capsize=4)
        ax.set_xlabel(hp_name)
        ax.set_ylabel("Loss finale")
        ax.set_title("Loss finale moyenne ± écart-type")
        ax.grid(True)
        if logy_loss:
            ax.set_yscale("log")
        if use_logx:
            ax.set_xscale("log")
        fig.savefig(f"figure_principale_{hp_name}.pdf")
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # Figure secondaire : historiques de loss
        # -----------------------------
        if show_loss_hist:
            plt.figure(figsize=(7, 4.5))

            for val, loss_histories in zip(vals, all_loss_histories):
                # On convertit chaque run en array
                arrs = [np.asarray(h, dtype=float) for h in loss_histories if len(h) > 0]
                if len(arrs) == 0:
                    continue

                # On tronque à la longueur minimale si jamais les runs diffèrent
                min_len = min(len(a) for a in arrs)
                arrs = np.array([a[:min_len] for a in arrs])

                mean_curve = arrs.mean(axis=0)
                std_curve = arrs.std(axis=0)
                epochs = np.arange(min_len)

                plt.plot(epochs, mean_curve, label=f"{hp_name}={val:g}")
                plt.fill_between(
                    epochs,
                    np.maximum(mean_curve - std_curve, 1e-16),
                    mean_curve + std_curve,
                    alpha=0.2
                )

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Historique de loss moyen — {hp_name}")
            plt.grid(True)
            if logy_loss:
                plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Historique_loss_{hp_name}.pdf", dpi = 300)
            plt.show()