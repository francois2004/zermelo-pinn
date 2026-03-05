"""
Définition d'une classe de NN suffisamment dérivables, et denses dans l'espace fonctionnel de résolution des 
EDP
"""

from torch import nn 


class PINN(nn.Module) : 
    """
    réseau de neurone entièrement connecté, activation tanh pour assurer la derivabilité
    entrée : (x,y) - coordonnées dans l'espace
    sortie : u(x,y) - valeur scalaire
    """
    def __init__(self, hidden_size = 64, n_layers = 4): 
        """
        Paramètres :
        hidden_size : int (64 par défaut pour assurer un bon equilibre entre vitesse et qualité de convergence)
        n_layers : int (4 par défaut pour assurer densité dans H2)

        """
        super().__init__()
        layers = [nn.Linear(2,hidden_size), nn.Tanh()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]

        self.network = nn.Sequential(*layers)

    def forward(self,x):
        """
        prends en entrées x (de dim 2) un tuple qui correspondent aux coordonnées dans l'espace, et renvoie
        la valeur associée par le NN 
        """
        return self.network(x)