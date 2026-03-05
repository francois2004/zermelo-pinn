""" Outils pour la définition de la perte informée par l'edp résolue"""
import torch
from src.models.pde import operateur_F
def derivatives_pinn(model, X): 
    """
    entrées : réseau de neurones, X un tuple de coordonnées spatiales
    renvoie les dérivées du nn en ce point X = (x,y)
    """
    X.requires_grad_(True)
    #print(X.shape)
    u = model(X)
    #print(u.shape)
    grads = torch.autograd.grad(u.sum(), X, create_graph = True)[0]
    ux = grads[:, 0:1]
    uy = grads[:, 1:2]

    uxx = torch.autograd.grad(ux.sum(), X, create_graph=True)[0][:, 0:1]

    uyy = torch.autograd.grad(uy.sum(), X, create_graph=True)[0][:, 1:2]

    return u, ux, uy, uxx, uyy

def loss_pde(model, xy_colloc, f_vals, vc_valls, params): 
    """
    prends en entrée le réseau de neurone et les points de collocations, renvoie
    la perte issue de l'EDP, qui est la norme 2 du résidu de l'opérateur différentiel lié a l'edp
    """
    u, ux, uy, uxx, uyy = derivatives_pinn(model, xy_colloc)

    return (operateur_F(ux, uy, uxx, uyy, vc_valls, params, torch)-f_vals).pow(2).mean()

def loss_inner(model, xy_inner): 
    """
    prend en entrée le nn, les points sur le cercle interieur, et renvoie 
    la pénalisation en norme 2, u = 0 a l'interieur
    """
    u = model(xy_inner)
    norm_u = u.pow(2).mean()
    return norm_u

def loss_outer(model, xy_outer):
    """
    prend en entrée le nn, les points sur le cercle exterieur, et renvoie 
    la pénalisation en norme 2, u = 1 a l'exterieur
    """
    u = model(xy_outer)
    norm_u = (u-1).pow(2).mean()
    return norm_u