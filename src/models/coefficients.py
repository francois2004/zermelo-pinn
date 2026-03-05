"""Module comprenant les coefficients de l'EDP que l'on cherche a résoudre"""

import numpy as np


def vc_field(x, y, params, xp = np):
    """
    Champ de vent perturbé : vc(x,y) = 1 - a·sin(π·s)
    où s = (‖(x,y)‖² - r²) / (R² - r²)
    """    
    s = ((x**2 + y**2)-params.r **2) / (params.R**2 - params.r**2)
    return 1.0 - params.a * xp.sin(xp.pi*s)

def u_exact(XX, YY, params, xp = np):
    """
    Solution exacte calculée sur une grille : 
    u(x, y) = sin(1/2 * pi * s)
    ou s = (‖(x,y)‖² - r²) / (R² - r²)
    """
    return xp.sin(0.5 * xp.pi * (XX**2 + YY**2 - params.r**2) / (params.R**2 - params.r**2))

def compute_f_exact(XX, YY, params,xp = np, hx=1e-5, hy=1e-5 ):
    """
    Calcul numérique du second membre f tel que u_exact soit solution de l'EDP.
    On évalue F(x, u, ∇u, D²u) avec les dérivées analytiques de u_exact.
   """
    phi = (XX**2 + YY**2 - params.r**2) / (params.R**2 - params.r**2)
    c   = 0.5 * xp.pi / (params.R**2 - params.r**2)
    # Dérivées de uex = sin(pi/2 * phi)
    ux  =  c * xp.cos(0.5*xp.pi*phi) * 2*XX
    uy  =  c * xp.cos(0.5*xp.pi*phi) * 2*YY
    uxx = -c**2 * xp.sin(0.5*xp.pi*phi) * (2*XX)**2 + c * xp.cos(0.5*xp.pi*phi) * 2
    uyy = -c**2 * xp.sin(0.5*xp.pi*phi) * (2*YY)**2 + c * xp.cos(0.5*xp.pi*phi) * 2

    norm2 = xp.sqrt(ux**2 + uy**2)
    norm1 = xp.abs(ux) + xp.abs(uy)

    VC = vc_field(XX,YY,params, xp)

    # f = -1/2 σx² uxx - 1/2 σy² uyy - vc*ux + vs*||∇u||_2 - κ*||∇u||_1
    f = (- 0.5*params.sig_x**2 * uxx
         - 0.5*params.sig_y**2 * uyy
         - VC * ux
         + params.vs * norm2
         - params.kappa * norm1)
    return f
    
def f_1(XX, YY, params, xp = np):
    """
    Calcule le second membre constant = 1
    """
    return xp.ones_like(XX)