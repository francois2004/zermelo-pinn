import numpy as np
from dataclasses import dataclass

@dataclass(frozen = True)
class Params: 
    r : float
    R : float
    kappa : float
    vs : float
    a : float
    sig_x : float
    sig_y : float
    X_min : float
    X_max : float
    Y_min : float
    Y_max : float

@dataclass(frozen = True)
class NumericalParams: 
    M : float #taille de la grille
    tol : float
    max_iter : int

@dataclass(frozen = True)
class PinnParams: 
    n_depth : int
    n_width : int

