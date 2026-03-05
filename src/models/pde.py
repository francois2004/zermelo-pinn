"""Module contenant les fonctions nécessaires à la création de l'operateur
 différentiel mies en cause dans l'equation de Zermelo"""

import numpy as np

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
    F += params.vs*xp.sqrt(ux**2 +uy**2) - params.kappa * (xp.abs(ux)+xp.abs(uy))
    return F
