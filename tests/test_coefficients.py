import numpy as np
import pytest
from config.parameters import Params
from src.models.coefficients import vc_field, compute_f_exact, f_1

# Instance de params partagée par tous les tests du fichier
params = Params(
    r=0.5, R=np.sqrt(2), kappa=0.1, vs=0.5,
    a=0.2, sig_x=0.5, sig_y=0.2,
    X_min=-2., X_max=2., Y_min=-2., Y_max=2.
)

def test_vc_field_shape():
    """La sortie a la même forme que les entrées."""
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    XX, YY = np.meshgrid(x, y)
    vc = vc_field(XX, YY, params)
    assert vc.shape == XX.shape

def test_vc_field_intern_values():
    """les valeurs au bord sont cohérentes"""
    theta = np.linspace(0, 2*np.pi, 100)
    XX = params.r *np.cos(theta)
    YY = params.r*np.sin(theta)
    vc = vc_field(XX, YY, params)
    assert np.allclose(vc, 1.0)

def test_vc_field_extern_values():
    """les valeurs au bord sont cohérentes"""
    theta = np.linspace(0, 2*np.pi, 100)
    XX = params.R *np.cos(theta)
    YY = params.R*np.sin(theta)
    vc = vc_field(XX, YY, params)
    assert np.allclose(vc, 1.0)

def test_f_1_shape():
    """La sortie a la même forme que les entrées."""
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    XX, YY = np.meshgrid(x, y)
    F_1 = f_1(XX, YY, params)
    assert F_1.shape == XX.shape    


def test_f_exacte_shape():
    """La sortie a la même forme que les entrées."""
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    XX, YY = np.meshgrid(x, y)
    F_1 = compute_f_exact(XX, YY, params)
    assert F_1.shape == XX.shape   