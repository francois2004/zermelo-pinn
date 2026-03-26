"""Module d'entrainement du PINN """
import numpy as np
import torch
import copy
import time
from src.models.pinn.loss import Loss
import src.models.coefficients as coeffs
device = torch.device("mps")

def sample_collocation(N, params, device = device):
    # Sur-échantillonnage fixe, une seule passe, zéro boucle while
    xy = (torch.rand(N * 6, 2, device=device) * 2*params.R) - params.R
    rad = torch.hypot(xy[:, 0], xy[:, 1])
    mask = (rad > params.r) & (rad < params.R)
    return xy[mask][:N]


def sample_boundary(N, params, device = device):
    """
    Tire N points sur ∂Br (bord intérieur) et N points sur ∂BR (bord extérieur),
    directement sur le device (CPU ou MPS).
    """
    theta_inner = 2 * torch.pi * torch.rand(N, device=device)
    theta_outer = 2 * torch.pi * torch.rand(N, device=device)

    xy_inner = torch.stack([
        params.r * torch.cos(theta_inner),
        params.r * torch.sin(theta_inner)
    ], dim=1)

    xy_outer = torch.stack([
        params.R * torch.cos(theta_outer),
        params.R * torch.sin(theta_outer)
    ], dim=1)

    return xy_inner, xy_outer


def train_SGD(model, params, f, N_colloc = 1000, N_bord = 1000, n_epochs = 1000, lam = .1, lr = 1e-3, tol = 1e-3, norme = 'l2', device = device): 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model = model.to(device)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    losses = []
    Time = 0
    #print("[train] génération du pool initial")
    X_colloc_pool = sample_collocation(N_colloc*10, params, device)
    X_inner_pool, X_outer_pool = sample_boundary(N_bord*10, params, device)
    for epoch in range(n_epochs):
        t0 = time.time()
        if epoch%50 == 0: 
            #print("[train] régénération du pool")
            X_colloc_pool = sample_collocation(N_colloc*10, params, device)
            X_inner_pool, X_outer_pool = sample_boundary(N_bord*10, params, device)

        idx_colloc = torch.randperm(N_colloc*10, device=device)[:N_colloc]
        idx_bord = torch.randperm(N_bord*10, device=device)[:N_colloc]

        xy_colloc = X_colloc_pool[idx_colloc]
        xy_inner = X_inner_pool[idx_bord]
        xy_outer = X_outer_pool[idx_bord]
        
        XX = xy_colloc[:, 0:1]
        YY = xy_colloc[:, 1:2]
        vc_vals = coeffs.vc_field(XX, YY, params, xp = torch)
        #print(vc_vals.shape)
        f_vals  = (f(XX, YY, params, xp = torch))
        ##Loss du modèle
        l_pde = Loss.loss_pde(model, xy_colloc, f_vals, vc_vals, params)
        l_bord = Loss.loss_bord(model, xy_inner, xy_outer, norme = norme)
        loss = l_pde + lam*l_bord
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        losses.append(loss.item())
        Time += (t1-t0)
        patience = n_epochs / 2
        if epoch > 2 * patience:
            mean_before = np.mean(losses[-2*patience : -patience])
            mean_after  = np.mean(losses[-patience:])
            decrease    = (mean_before - mean_after) / (mean_before + 1e-12)
            if decrease < tol:
                print(f"  Early stopping à l'epoch {epoch} "
                      f"(amélioration relative = {decrease:.2e} < {tol})")
                break
    return losses, Time

def train_policy_step(
    model,
    model_policy,
    params,
    vc_field,
    f,
    eta_k,
    eta0=1.0,
    min_epochs=200,
    max_epochs=3000,
    lr=1e-3,
    optimizer=None,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    verbose=True,
    N_colloc=1000,
    device=None,
):
    """
    Résout approximativement le sous-problème linéarisé gelé associé à model_policy.

    Critère d'arrêt inspiré du papier :
        J_k(u_{k+1}) <= eta_k * min( ||u_{k+1}-u_k||_{H2,h}^2 , eta0 )
    """
    history = {
        "loss": [],
        "loss_pde": [],
        "loss_bnd": [],
        "h2_dist": [],
        "target": [],
    }

    if device is None:
        device = next(model.parameters()).device

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model_policy.eval()

    # === collocation fixe pour toute l’itération de policy ===
    X_colloc = sample_collocation(N_colloc, params, device)
    X_inner, X_outer = sample_boundary(N_colloc, params, device)

    XX = X_colloc[:, 0:1]
    YY = X_colloc[:, 1:2]

    vc_vals = vc_field(XX, YY, params, xp=torch)
    f_vals = f(XX, YY, params, xp=torch)

    # === contrôles gelés UNE FOIS à partir de model_policy ===
    _, ux_pol, uy_pol, _, _ = Loss.derivatives_pinn(model_policy, X_colloc)
    with torch.no_grad():
        alpha, beta_x, beta_y = Loss.compute_feedback_control(ux_pol, uy_pol, params)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        l_pde = Loss.loss_Linearized_PDE(
            model, X_colloc, alpha, beta_x, beta_y, params, vc_vals, f_vals
        )
        l_bord = Loss.loss_bord(
            model, X_inner, X_outer, norme=boundary_mode
        )
        loss = l_pde + gamma_boundary * l_bord

        loss.backward()
        optimizer.step()

        # distance discrète H² entre u_{k+1} courant et u_k gelé
        h2_dist = policy_iteration_error(model, model_policy, X_colloc)
        target = eta_k * min(h2_dist ** 2, eta0)
        target = min(target, 1.0)

        history["loss"].append(loss.item())
        history["loss_pde"].append(l_pde.item())
        history["loss_bnd"].append(l_bord.item())
        history["h2_dist"].append(h2_dist)
        history["target"].append(target)

        if verbose and epoch % 200 == 0:
            print(
                f"epoch {epoch:5d} | "
                f"loss={loss.item():.3e} | "
                f"pde={l_pde.item():.3e} | "
                f"bnd={l_bord.item():.3e} | "
                f"H2={h2_dist:.3e} | "
                f"target={target:.3e}"
            )

        # critère d'arrêt interne
        if epoch >= min_epochs and loss.item() <= target:
            if verbose:
                print(f"[inner stop] epoch={epoch} | loss={loss.item():.3e} <= target={target:.3e}")
            break

    return model, history

def policy_iteration_error(model_a, model_b, X, batch_size = 256):
    errs = []
    for i in range(0, X.shape[0], batch_size):
        Xb = X[i:i + batch_size].clone().detach().to(device)

        ua, uxa, uya, uxxa, uyya = Loss.derivatives_pinn(model_a, Xb)
        ub, uxb, uyb, uxxb, uyyb = Loss.derivatives_pinn(model_b, Xb)

        err2 = (
            (ua - ub).pow(2).mean()
            + (uxa - uxb).pow(2).mean()
            + (uya - uyb).pow(2).mean()
            + (uxxa - uxxb).pow(2).mean()
            + (uyya - uyyb).pow(2).mean()
        )

        errs.append(err2.detach())

        del Xb, ua, uxa, uya, uxxa, uyya, ub, uxb, uyb, uxxb, uyyb, err2

    
    return torch.sqrt(torch.stack(errs).mean()).item()   


def train_policy_iteration(
    model,
    params,
    vc_field,
    f,
    n_policy_iter=10,
    lr=1e-3,
    tol_policy=1e-4,
    xy_eval=None,
    boundary_mode="H1_2",
    gamma_boundary=1.0,
    verbose=True,
    N_colloc=1000,
    eta0=1.0,
    eta_schedule=None,
    min_epochs=200,
    max_epochs=3000,
    keep_best=True,
):
    """
    Boucle externe de policy iteration.

    On résout chaque sous-problème avec un critère d'arrêt dépendant de eta_k.
    """
    history_policy = []
    device = next(model.parameters()).device

    if xy_eval is None:
        xy_eval = sample_collocation(5000, params, device)

    if eta_schedule is None:
        eta_schedule = [0.5 ** k for k in range(n_policy_iter)]

    best_model = copy.deepcopy(model)
    best_err = float("inf")

    for k in range(n_policy_iter):
        eta_k = eta_schedule[k]

        if verbose:
            print(f"\n=== Policy iteration {k} | eta_k={eta_k:.3e} ===")


        model_old = copy.deepcopy(model)
        model_old.eval()

        model, history_inner = train_policy_step(
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
            verbose=verbose,
            N_colloc=N_colloc,
            device=device,
        )

        err_iter = policy_iteration_error(model_old, model, xy_eval)

        history_policy.append({
            "policy_iter": k,
            "eta_k": eta_k,
            "err_iter": err_iter,
            "history_inner": history_inner,
        })

        if verbose:
            print(f"policy H2-error = {err_iter:.3e}")

        # garder le meilleur itéré au sens de la policy error
        if keep_best and err_iter < best_err:
            best_err = err_iter
            best_model = copy.deepcopy(model)

        if err_iter < tol_policy:
            if verbose:
                print("Convergence de la policy iteration atteinte.")
            break

    if keep_best:
        return best_model, history_policy
    return model, history_policy