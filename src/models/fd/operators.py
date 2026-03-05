"""
Module comprenant le calcul des opérateurs à différences finies
implémentés en sparse pour améliorer le temps de calcul
"""
import numpy as np
import scipy.sparse as sparse 

## Différences finies: 
def D_1_X_sparse(Nx,Ny, params): 
    """
    Matrice de différences finies centrées d'ordre 1 en x.
    Taille : (Nx+1)(Ny+1) × (Nx+1)(Ny+1).
    Les connexions entre lignes de grille sont coupées.
    """
    h = (params.X_max - params.X_min)/Nx
    N = (Nx+1)*(Ny+1)
    diag_p = np.ones(N - 1)
    diag_m = np.ones(N - 1)
    for j in range(Ny + 1):
        idx = j * (Nx + 1) + Nx   
        if idx < N - 1:
            diag_p[idx] = 0       
    for j in range(Ny + 1):
        idx = j * (Nx + 1)        
        if idx > 0:
            diag_m[idx - 1] = 0 
    D = sparse.diags(diagonals = [-diag_m, diag_p], offsets = [-1,1], format = 'csr')/(2*h)
    D = D.tocsr()
    return D

def D_1_Y_sparse(Nx, Ny, params): 
    """
    Matrice de différences finies centrées d'ordre 1 en y.
    Taille : (Nx+1)(Ny+1) × (Nx+1)(Ny+1).
    """
    k = (params.Y_max - params.Y_min)/Ny
    vect_0 = np.ones((Nx + 1) * (Ny + 1))
    D = sparse.diags(diagonals = [-vect_0[Nx+1:], vect_0[:-Nx-1]], offsets = [-Nx-1, Nx+1], format = 'csr') / (2*k)
    D = D.tocsr()
    return D


def D_2_X_sparse(Nx, Ny, params): 
    """
    Matrice de différences finies centrées d'ordre 2 en x.
    Taille : (Nx+1)(Ny+1) × (Nx+1)(Ny+1).
    Les connexions entre lignes de grille sont coupées.
    """
    h = (params.X_max - params.X_min)/Nx
    N = (Nx + 1) * (Ny + 1)
    
    diag_0  = -2 * np.ones(N)
    diag_p =       np.ones(N - 1)
    diag_m =       np.ones(N - 1)
    for j in range(Ny + 1):
        idx = j * (Nx + 1) + Nx   
        if idx < N - 1:
            diag_p[idx] = 0       
    for j in range(Ny + 1):
        idx = j * (Nx + 1)        
        if idx > 0:
            diag_m[idx - 1] = 0 
    D = sparse.diags(diagonals = [diag_m, diag_0, diag_p], offsets = [-1, 0, 1], format = 'csr') / h**2
    D = D.tocsr()
    return D

def D_2_Y_sparse(Nx, Ny, params):
    """
    Matrice de différences finies centrées d'ordre 2 en y.
    Taille : (Nx+1)(Ny+1) × (Nx+1)(Ny+1).
    """
    k = (params.Y_max - params.Y_min)/Ny
    N = (Nx + 1) * (Ny + 1)
    
    diag_0  = -2 * np.ones(N)
    diag_p =      np.ones(N - (Nx + 1))
    diag_m =      np.ones(N - (Nx + 1))

    D = sparse.diags(diagonals = [diag_p, diag_0, diag_m], offsets = [-Nx-1, 0, Nx+1], format = 'csr') / k**2
    D = D.tocsr()
    return D

def alpha(ux, uy ): 
    """
    Direction normalisée du gradient : (ux, uy) / ‖∇u‖₂.
    Renvoie (0, 0) si le gradient est nul.
    """
    norm = np.sqrt((ux)**2 + (uy)**2)
    norm = np.where(norm == 0, 1, norm)
    return ux/norm, uy/norm

def beta(ux, uy): 
    """
    Signe du gradient : (sign(ux), sign(uy)).
    Approxime la direction de ‖∇u‖₁.
    """
    return np.sign(ux), np.sign(uy)

def build_fd_operators(Nx, Ny, params):
    """
    Construit et retourne tous les opérateurs en une seule passe.
    À appeler une fois et stocker le résultat — évite les reconstructions.

    Returns
    -------
    dict avec clés 'Dx', 'Dy', 'Dxx', 'Dyy'
    """
    return {
        "Dx":  D_1_X_sparse(Nx, Ny, params),
        "Dy":  D_1_Y_sparse(Nx, Ny, params),
        "Dxx": D_2_X_sparse(Nx, Ny, params),
        "Dyy": D_2_Y_sparse(Nx, Ny, params),
    }