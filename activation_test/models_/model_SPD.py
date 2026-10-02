
import math
import torch
import torch.nn as nn
from spd_learn.modules import BiMap, LogEig, ReEig

from activation_test.activation.spectral import PowerEig, SpAEig, TanhEig
from activation_test.activation.elementwise import activationSPD, coshP, coshPTraceNorm, polynomialActivation, sinhP, expT, expP


GOLDEN = (1 + math.sqrt(5)) / 2


def compute_dims(n_chans, division = "half", depth = 1, n_min = 9):
    """ 
    Dimensions [n_0, n_1, ..., n_L] of the SPD block.

    division = "half"   : n_l = n_0 // 2**l, for l = 1..depth
    division = "golden" : n_l = round(n_{l-1} / phi), as long as n_l >= n_min (depth is ignored)
    """
    if division == "half":
        dims = [n_chans // 2**l for l in range(depth + 1)]
    elif division == "golden":
        dims = [n_chans]
        while round(dims[-1] / GOLDEN) >= n_min:
            dims.append(round(dims[-1] / GOLDEN))
    else:
        raise ValueError(f"Unknown division : {division}")

    if len(dims) < 2 or dims[-1] < 1:
        raise ValueError(f"No valid SPD block for n_chans={n_chans}, division={division}, depth={depth} : {dims}")
    return dims


class modelSPDNet(nn.Module): 

    def __init__(self, activation = "reeig", division = "half", depth = 1, n_min = 9, threshold = 1e-4, n_chans = None, domains = None, upper = True, n_outputs = None):
        super().__init__()
        
        if n_chans is None : 
            raise ValueError("n_chans must be provided")
        if domains is None :                                                
            raise ValueError("domains must be provided")
        
        self.activation_type = activation
        self.threshold = threshold

        # -------------------------------------
        # Dimensions of the BiMap layers (saved with the results)
        # -------------------------------------
        self.dims = compute_dims(n_chans, division, depth, n_min)

        # -------------------------------------
        # domain dependent architecture : [BiMap -> activation] x L
        # -------------------------------------
        self.domains_block = nn.ModuleDict()                                   # Domain specific 
        for domain in domains : 
            layers = {}
            for l in range(1, len(self.dims)):
                layers[f"bimap{l}"] = BiMap(self.dims[l - 1], self.dims[l])
                layers[f"activation{l}"] = self._make_activation(n = self.dims[l])
            self.domains_block[domain] = nn.ModuleDict(layers)

        # -------------------------------------
        # Logeig and linear layer 
        # -------------------------------------
        last_dim = self.dims[-1]
        self.logeig = LogEig(upper = upper)                                     # if Upper = True : vech                    
        self.len_last_layer = (
            last_dim * (last_dim + 1) // 2 if upper else last_dim**2
        )

        self.classifier = nn.Linear(self.len_last_layer, n_outputs)             # Linear layer


    def _make_activation(self, n):

        if self.activation_type == "reeig":
            return ReEig(self.threshold)

        elif self.activation_type == "powereig":
            return PowerEig(threshold=self.threshold)

        elif self.activation_type == "tanheig":
            return TanhEig(threshold=self.threshold)
        
        elif self.activation_type == "spaeig":
            return SpAEig(threshold=self.threshold)

        elif self.activation_type == "sinh":
            return activationSPD(mode="sinh")

        elif self.activation_type == "cosh":
            return activationSPD(mode="cosh")

        elif self.activation_type == "exp":
            return activationSPD(mode="exp")
        
        elif self.activation_type == "coshP":
            return coshP()

        elif self.activation_type == "coshPnorm":
            return coshPTraceNorm()

        elif self.activation_type == "sinhP":
            return sinhP()
        
        elif self.activation_type == "polyact":
            return polynomialActivation()

        elif self.activation_type == "expT":
            return expT()

        elif self.activation_type == "expP":
            return expP()

        else:
            raise ValueError("Unknown activation")


    def forward(self, X: torch.Tensor, domain) -> torch.Tensor:
        """
        Forward pass of the SPDNet model.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. 

        Returns
        -------
        torch.Tensor
            Output of the classifier, with shape `(batch_size, n_outputs)`.
        """

        block = self.domains_block[domain]
        for layer in block.values():
            X = layer(X)
        X = self.logeig(X)
        X = self.classifier(X)

        return X 
