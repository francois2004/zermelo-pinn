"""Module contenant les fonctions nécessaires à la création de l'operateur
 différentiel mies en cause dans l'equation de Zermelo"""

import numpy as np
import torch
def operateur_F( ux, uy, uxx, uyy, vc, params, xp = np): 
    """
    prends en entrée les dérivées , et 
    renvoie l'opérateur différentiel
    F(ux,uy,uxx,uyy) = -1/2 Sigma^2 D_2 -vc*ux + vs*||grad(u)||_2 - kappa*||grad(u)||_1
    """
    ## terme de second ordre
    F = -1/2*( params.sig_x**2 * uxx + params.sig_y**2 * uyy)
    ## terme en vc
    F += -vc * ux
    ## termes de norme
    F += params.vs*xp.sqrt(ux**2 +uy**2) - params.kappa * (params.sig_x* xp.abs(ux) + params.sig_y* xp.abs(uy))
    return F

def operateur_F_linearise(ux, uy, uxx, uyy, vc, alpha, bet_x, bet_y, params): 
    """
    prends en entrée les dérivées , et 
    renvoie l'opérateur différentiel linéarisé
    F(ux,uy,uxx,uyy) = -1/2 Sigma^2 D_2 -vc*ux + vs*(cos(alpha)*ux + sin(alpha)uy) - kappa*(bet_x*sig_x *u_x + bet_y...
    """
    ## terme de second ordre
    F = -1/2*( params.sig_x**2 * uxx + params.sig_y**2 * uyy)
    ## terme en vc
    F += -vc * ux
    ##terme linéarisé en alpha
    F += params.vs*(torch.cos(alpha)*ux +  torch.sin(alpha)*uy)
    ##terme linéarisé en beta
    F = F -(bet_x*params.sig_x*ux + bet_y*params.sig_y*uy )
    return F