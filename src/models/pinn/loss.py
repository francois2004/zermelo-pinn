""" Outils pour la définition de la perte informée par l'edp résolue"""
import torch
from src.models.pde import operateur_F, operateur_F_linearise
class Loss : 
    @staticmethod
    def derivatives_pinn(model, X): 
        """
        entrées : réseau de neurones, X un tuple de coordonnées spatiales
        renvoie les dérivées du nn en ce point X = (x,y)
        """
        X = X.clone().detach().requires_grad_(True)
        #print(X.shape)
        u = model(X)
        #print(u.shape)
        grads = torch.autograd.grad(u.sum(), X, create_graph = True)[0]
        ux = grads[:, 0:1]
        uy = grads[:, 1:2]

        uxx = torch.autograd.grad(ux.sum(), X, create_graph=True)[0][:, 0:1]

        uyy = torch.autograd.grad(uy.sum(), X, create_graph=True)[0][:, 1:2]

        return u, ux, uy, uxx, uyy
    @staticmethod
    def loss_pde(model, xy_colloc, f_vals, vc_valls, params): 
        """
        prends en entrée le réseau de neurone et les points de collocations, renvoie
        la perte issue de l'EDP, qui est la norme 2 du résidu de l'opérateur différentiel lié a l'edp
        """
        u, ux, uy, uxx, uyy = Loss.derivatives_pinn(model, xy_colloc)

        return (operateur_F(ux, uy, uxx, uyy, vc_valls, params, torch)-f_vals).pow(2).mean()

    
    @staticmethod
    def cond_l2(model, xy_bord, target):
        residual = model(xy_bord)- target
        return residual.pow(2).mean()
    @staticmethod  
    def cond_H_1_2(model, xy_bord, target): 
        residual = (model(xy_bord)-target).reshape(-1)
        loss = residual.pow(2).mean()
        R = residual[None, :] - residual[:, None]
        x = xy_bord[None,:,:] - xy_bord[:,None,:]
        x_norm = x.pow(2).sum(dim = -1)
        x_norm = torch.clamp(x_norm, min = 1e-8)
        loss+= (R.pow(2)/(x_norm)).mean()
        return loss
    @staticmethod
    def cond_H_3_2(model, xy_bord, target):
        u, ux, uy, _, _ = Loss.derivatives_pinn(model, xy_bord)
        residual = u - target
        loss = residual.pow(2).mean() 
        x, y = xy_bord[: , 0:1], xy_bord[:,1:2]
        norm = torch.sqrt(x.pow(2)+y.pow(2)+1e-8)
        nx, ny = -y/ norm, x/norm
        un = (ux*nx + uy * ny).reshape(-1)
        loss += un.pow(2).mean()
        R = un[None, :] - un[:, None]
        x = xy_bord[None,:,:] - xy_bord[:,None,:]
        x_norm = x.pow(2).sum(dim = -1)
        x_norm = torch.clamp(x_norm, min = 1e-8)
        loss += (R.pow(2)/x_norm).mean()
        return loss
    
    @staticmethod
    def loss_bord(model, xy_inner, xy_outer, norme='l2'):
        """
    Pénalisation des conditions aux bords :
        - bord intérieur : u = 0
        - bord extérieur : u = 1
    
    Paramètres
    ----------
    norme : 'l2', 'H1_2', 'H3_2'
        """
    # ── sélection de la norme ────────────────────
        norme_dict = {
        'l2'   : Loss.cond_l2,
        'H1_2' : Loss.cond_H_1_2,
        'H3_2' : Loss.cond_H_3_2,
        }
    
        if norme not in norme_dict:
            raise ValueError(f"Norme '{norme}' inconnue. Choisir parmi {list(norme_dict.keys())}")
    
        cond = norme_dict[norme]
    
        # ── targets ──────────────────────────────────
        target_inner = torch.zeros(xy_inner.shape[0], 1, device=xy_inner.device)
        target_outer = torch.ones( xy_outer.shape[0], 1, device=xy_outer.device)

        # ── somme des deux pertes ─────────────────────
        return cond(model, xy_inner, target_inner) + cond(model, xy_outer, target_outer)

    @staticmethod
    def compute_feedback_control(ux, uy, params):
        """
        Construis les contrôles optimaux gelés 
        """
        b_x = params.kappa*torch.sgn(params.sig_x*ux)
        b_y = params.kappa*torch.sgn(params.sig_y*uy)
        alp = torch.atan2(uy, ux)
        return (alp, b_x, b_y)

    @staticmethod
    def loss_Linearized_PDE(model, xy, alp, bx, by, params, vc_vals, f_vals):
        """
        Calcule le résidu de la PDE linéarisée.
        """
        _, ux, uy, uxx, uyy = Loss.derivatives_pinn(model, xy)
        F = operateur_F_linearise(ux, uy, uxx, uyy, vc_vals, alp, bx, by, params)
        return (F - f_vals).pow(2).mean()
    




        







