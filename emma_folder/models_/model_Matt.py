import torch
import torch.nn as nn

# L'importation clé : on récupère le bloc d'attention et LogEig
from spd_learn.models.matt import AttentionManifold 
from spd_learn.modules import LogEig

# Vos importations d'activations
from emma_folder.activation.spectral import PowerEig, SpAEig, TanhEig
from emma_folder.activation.elementwise import activationSPD, coshP, polynomialActivation, sinhP, expT, expP
from spd_learn.modules import BiMap, ReEig, LogEig


class modelMAtt_Custom(nn.Module): 
    def __init__(self, activation="reeig", n_chans=None, domains=None, n_outputs=None, threshold=1e-4):
        super().__init__()
        
        if n_chans is None: raise ValueError("n_chans must be provided")
        if domains is None: raise ValueError("domains must be provided")
        if n_outputs is None: raise ValueError("n_outputs must be provided")
        
        self.activation_type = activation
        self.threshold = threshold
        
        # Le modèle MAtt réduit généralement légèrement la dimension lors de l'attention
        # On définit une taille de sortie (par exemple 80% des canaux initiaux)
        out_features = int(n_chans * 0.8) 
        if out_features < 2: out_features = 2 # Sécurité

        # -------------------------------------
        # Architecture dépendante du domaine
        # -------------------------------------
        self.domains_block = nn.ModuleDict()                                   
        for domain in domains: 
            layers = {
                # 1. Le mécanisme d'attention géométrique remplace la BiMap
                "attention": AttentionManifold(in_features=n_chans, out_features=out_features),
                
                # 2. Votre fonction d'activation personnalisée
                "activation1": self._make_activation(n=out_features),
            }
            self.domains_block[domain] = nn.ModuleDict(layers)

        # -------------------------------------
        # Logeig et classification
        # -------------------------------------
        self.logeig = LogEig(upper=True)                                    
        self.len_last_layer = out_features * (out_features + 1) // 2

        self.classifier = nn.Linear(self.len_last_layer, n_outputs)             


    def _make_activation(self, n):
        # ... Exactement le même code que dans votre fichier model_SPD.py ...
        if self.activation_type == "reeig": return ReEig(self.threshold)
        elif self.activation_type == "coshP": return coshP()
        elif self.activation_type == "expT": return expT()
        # Ajoutez vos autres activations ici...
        else: raise ValueError("Unknown activation")


    def forward(self, X: torch.Tensor, domain) -> torch.Tensor:
        """
        X entre avec la forme : (batch_size, 64, 64)
        """
        # L'ASTUCE POUR MATT : On ajoute la dimension "patch"
        X = X.unsqueeze(1) # X devient (batch_size, 1, 64, 64)

        block = self.domains_block[domain]
        for layer in block.values():
            X = layer(X)
            
        # L'attention renvoie (batch, 1, out_features, out_features)
        # On retire la dimension "patch" pour repasser sur du classique
        X = X.squeeze(1) # X redevient (batch, out_features, out_features)

        # Passage en géométrie euclidienne et classification
        X = self.logeig(X)
        X = self.classifier(X)

        return X