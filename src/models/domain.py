"""Domaine"""
import numpy as np


def make_masks(X, Y,params):
    """
    prends en entrées les paramètres du problème ainsi que le nombre de points, 
    renvoie trois masques sur la matrice qui nous dit sin on est dans la région du problème ou non
    """
    rad = np.sqrt(X**2 + Y**2)
    mask_0 = (rad <= params.r)
    mask_1 = (rad >= params.R)
    mask_PDE = (~mask_0) & (~mask_1)
    return mask_0, mask_PDE, mask_1