"""
Tests unitaires pour src/fd/operators.py
Vérifie les dimensions des matrices sparse et le comportement
de alpha et beta.
"""

import numpy as np
import pytest
from config.parameters import Params
from src.models.fd.operators import (
    D_1_X_sparse, D_1_Y_sparse,
    D_2_X_sparse, D_2_Y_sparse,
    alpha, beta
)

params = Params(
    r=0.5, R=np.sqrt(2), kappa=0.1, vs=0.5,
    a=0.2, sig_x=0.5, sig_y=0.2,
    X_min=-2., X_max=2., Y_min=-2., Y_max=2.
)

N = 20  # taille de grille utilisée dans tous les tests
SIZE = (N + 1) ** 2

# ------------------------------------------------------------------ D_1_X

def test_D_1_X_shape():
    """D_1_X doit être une matrice carrée de taille (N+1)²."""
    D = D_1_X_sparse(N, N, params)
    assert D.shape == (SIZE, SIZE)

def test_D_1_X_derivative():
    """D_1_X appliqué à f(x,y)=x doit approcher df/dx=1 à l'intérieur."""
    x = np.linspace(params.X_min, params.X_max, N + 1)
    y = np.linspace(params.Y_min, params.Y_max, N + 1)
    XX, YY = np.meshgrid(x, y)
    U = XX.flatten()
    D = D_1_X_sparse(N, N, params)
    dU = D @ U
    # Reconstruire en 2D pour sélectionner proprement
    dU_2D = dU.reshape(N + 1, N + 1)
    # Exclure la première et dernière colonne (bords en x)
    interior_2D = dU_2D[:, 1:-1]
    assert np.allclose(interior_2D, 1.0, atol=1e-10)

# ------------------------------------------------------------------ D_1_Y

def test_D_1_Y_shape():
    """D_1_Y doit être une matrice carrée de taille (N+1)²."""
    D = D_1_Y_sparse(N, N, params)
    assert D.shape == (SIZE, SIZE)

def test_D_1_Y_derivative():
    """D_1_Y appliqué à f(x,y)=y doit approcher df/dy=1 à l'intérieur."""
    x = np.linspace(params.X_min, params.X_max, N + 1)
    y = np.linspace(params.Y_min, params.Y_max, N + 1)
    XX, YY = np.meshgrid(x, y)
    U = YY.flatten()
    D = D_1_Y_sparse(N, N, params)
    dU = D @ U
    # Reconstruire en 2D pour sélectionner proprement
    dU_2D = dU.reshape(N + 1, N + 1)
    # Exclure la première et dernière colonne (bords en x)
    interior_2D = dU_2D[:, 1:-1]
    assert np.allclose(interior_2D, 1.0, atol=1e-10)

# ------------------------------------------------------------------ D_2_X / D_2_Y

def test_D_2_X_shape():
    """D_2_X doit être une matrice carrée de taille (N+1)²."""
    D = D_2_X_sparse(N, N, params)
    assert D.shape == (SIZE, SIZE)

def test_D_2_Y_shape():
    """D_2_Y doit être une matrice carrée de taille (N+1)²."""
    D = D_2_Y_sparse(N, N, params)
    assert D.shape == (SIZE, SIZE)

def test_D_2_X_derivative():
    """D_2_X appliqué à f(x,y)=x² doit approcher d²f/dx²=2 à l'intérieur."""
    x = np.linspace(params.X_min, params.X_max, N + 1)
    y = np.linspace(params.Y_min, params.Y_max, N + 1)
    XX, YY = np.meshgrid(x, y)
    U = (XX ** 2).flatten()
    D = D_2_X_sparse(N, N, params)
    dU = D @ U
    interior = np.arange(N + 2, SIZE - N - 2)
    assert np.allclose(dU[interior], 2.0, atol=1e-6)

# ------------------------------------------------------------------ alpha

def test_alpha_shape():
    """alpha doit retourner deux vecteurs de taille (N+1)²."""
    ux = np.random.randn(SIZE)
    uy = np.random.randn(SIZE)
    ax, ay = alpha(ux, uy)
    assert ax.shape == (SIZE,)
    assert ay.shape == (SIZE,)

def test_alpha_unit_norm():
    """Le vecteur (ax, ay) doit être de norme 1 quand le gradient est non nul."""
    ux = np.random.randn(SIZE) + 1.0  # +1 pour éviter les zéros
    uy = np.random.randn(SIZE) + 1.0
    ax, ay = alpha(ux, uy)
    norm = np.sqrt(ax ** 2 + ay ** 2)
    assert np.allclose(norm, 1.0, atol=1e-10)

def test_alpha_zero_gradient():
    """alpha doit retourner (0, 0) quand le gradient est nul."""
    ux = np.zeros(SIZE)
    uy = np.zeros(SIZE)
    ax, ay = alpha(ux, uy)
    assert np.allclose(ax, 0.0)
    assert np.allclose(ay, 0.0)

# ------------------------------------------------------------------ beta

def test_beta_shape():
    """beta doit retourner deux vecteurs de taille (N+1)²."""
    ux = np.random.randn(SIZE)
    uy = np.random.randn(SIZE)
    bx, by = beta(ux, uy)
    assert bx.shape == (SIZE,)
    assert by.shape == (SIZE,)

def test_beta_values():
    """beta doit retourner uniquement des valeurs dans {-1, 0, 1}."""
    ux = np.random.randn(SIZE)
    uy = np.random.randn(SIZE)
    bx, by = beta(ux, uy)
    assert np.all(np.isin(bx, [-1, 0, 1]))
    assert np.all(np.isin(by, [-1, 0, 1]))