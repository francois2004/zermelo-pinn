"""
Contient la fonction qui construit le système linéaire engendré par la discrétisation par différences finies de l'equation, 
ainsi qu'un solveur se basant sur la méthode de Newton pour résoudre le problème d'optimisation
"""

import numpy as np
import scipy.sparse.linalg as lng
import scipy.sparse as sparse
import src.models.fd.operators as ops
import src.models.coefficients as coeffs
from src.models.domain import make_masks

def build_system(XX, YY, U, N, f, params, fd_ops) :
    """
    Prends en entrées U, le pas de discrétisation N, les coefficients vc et f, 
    et renvoie les coeffficients du système linéaire résolu dans le solveur
    """
    ## pour etre sur de traiter avec la forme dont on a besoin
    U = U.flatten()
    
    ##matrices de différences finies
    ux = fd_ops['Dx']
    uy = fd_ops['Dy']
    uxx = fd_ops['Dxx']
    uyy = fd_ops['Dyy']

    ## matrice et coefficients a partir du U
    ax, ay = ops.alpha(ux@U, uy@U)
    bx, by = ops.beta(ux@U, uy@U)

    vc = coeffs.vc_field(XX, YY, params)
    vc = vc.flatten()
    #print(VC.shape)
    ## ordre 2
    A = (-.5*params.sig_x**2 * uxx -.5*params.sig_y**2 * uyy) 
    ## vents : 
    A = A - sparse.diags(vc) @ ux
    ## termes en alpha
    A = A + params.vs * (sparse.diags(ax)@ux + sparse.diags(ay)@uy)
    ## termes en beta
    A = A - params.kappa * (sparse.diags(bx) @ ux + sparse.diags(by)@uy)

    ## second membre 
    B = f(XX,YY, params).flatten()

    m1, m2, m3 = make_masks(XX, YY, params)
    m1 = m1.flatten()
    m2 = m2.flatten()
    m3 = m3.flatten()
    ## conditions aux limites :
    bd_idx  = np.where(~m2)[0]
    int_idx = np.where(m2)[0]

    P_in = sparse.diags(m2.astype(float), format="csr")
    P_bd = sparse.diags((~m2).astype(float), format="csr")
    A = P_in @ A + P_bd        

    B[bd_idx] = np.where(m1[bd_idx], 0.0, 1.0)
    
    return A, B

def Solveur_Zermelo (N, f,params, tol = 1e-6, max_iter = 100):
    """
    prends en entrées les paramètres du problème, la grille, le fonction, et 
    renvoie la solution approchée par méthode de Newton
    """
    xx, yy =np.linspace(params.X_min, params.X_max, N+1), np.linspace(params.Y_min, params.Y_max, N+1 )
    XX, YY = np.meshgrid(xx, yy)
    mask_in, mask_pde, mask_out = make_masks(XX, YY, params)

    U_vec = np.zeros((N+1)*(N+1))
    ##initialiser les bords : 
    mask_out = mask_out.flatten()
    U_vec[mask_out] = 1.0
    ops_dict = ops.build_fd_operators(N, N, params)

    for n in range(max_iter):
        A,B = build_system( XX, YY,U_vec, N,f, params, ops_dict)
        U_next = lng.spsolve(A,B)

        err = np.max(np.abs(U_next - U_vec))
        U_vec = U_next
        if err < tol: 
            break
    U = U_vec.reshape(N+1, N+1)
    U[~mask_pde] = np.nan
    return U, XX, YY 