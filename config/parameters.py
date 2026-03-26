import numpy as np
from dataclasses import dataclass

from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    r :float = 0.5
    R :float = np.sqrt(2)
    kappa :float = .1
    vs :float = .5
    a :float = .2
    sig_x :float= .5
    sig_y :float = .2
    X_min :float = -2.0
    X_max :float = 2.0
    Y_min :float = -2.0
    Y_max :float = 2.0

@dataclass(frozen = True)
class NumericalParams: 
    M : float #taille de la grille
    tol : float
    max_iter : int

@dataclass(frozen = True)
class PinnParams: 
    n_depth : int
    n_width : int

