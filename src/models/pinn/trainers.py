"""Module d'entrainement du PINN """
import numpy as np
import torch
import time
from src.models.pinn.loss import loss_inner, loss_outer, loss_pde
import src.models.coefficients as coeffs


def sample_collocation(N, params):
    # Sur-échantillonnage fixe, une seule passe, zéro boucle while
    xy = np.random.uniform(-params.R, params.R, size=(N * 6, 2))
    rad = np.hypot(xy[:, 0], xy[:, 1])
    mask = (rad > params.r) & (rad < params.R)
    return torch.tensor(xy[mask][:N], dtype=torch.float32)


def sample_boundary(N, params):
    """
    Tire N points sur ∂Br (bord intérieur) et N points sur ∂BR (bord extérieur).
    """
    theta = np.random.uniform(0, 2 * np.pi, size=N)

    # Bord intérieur : rayon r
    xy_inner = np.stack([params.r * np.cos(theta),
                         params.r * np.sin(theta)], axis=1)

    # Bord extérieur : rayon R
    xy_outer = np.stack([params.R * np.cos(theta),
                         params.R * np.sin(theta)], axis=1)

    return (torch.tensor(xy_inner, dtype=torch.float32),
            torch.tensor(xy_outer, dtype=torch.float32))


def train(model, params, f, N_colloc = 10000, N_bord = 1000, n_epochs = 1000, lam = 5., lr = 1e-3, tol = 1e-3): 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    losses = []
    Time = 0
    for epoch in range(n_epochs):
        t0 = time.time()
        xy_colloc = sample_collocation(N_colloc, params).to(device)
        xy_inner, xy_outer = sample_boundary(N_bord, params)
        xy_inner = xy_inner.to(device)
        xy_outer = xy_outer.to(device)
        
        XX = xy_colloc[:, 0:1]
        YY = xy_colloc[:, 1:2]
        vc_vals = coeffs.vc_field(XX, YY, params, xp = torch)
        #print(vc_vals.shape)
        f_vals  = (f(XX, YY, params, xp = torch))
        ##Loss du modèle
        l_pde = loss_pde(model, xy_colloc, f_vals, vc_vals, params)
        l_bord = loss_inner(model, xy_inner) + loss_outer(model, xy_outer)
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