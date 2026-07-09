import os
import sys

import torch

from spd_learn.modules import BiMap
from activation_test.activation.elementwise import coshP
from preprocessing.data_scripts.get_eeg_data import load_covmats

DEVICE = "cpu"
DTYPE = torch.float64

# La matrice de test
DATA_DIR = r"C:/Users/alexc/Desktop/internship/data/MatCov"
DB_PREFIX = "BNCI2014001"
FILE_ID = 1


# =============================================================================
# LEs FONCTIOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONS
# =============================================================================
def trace(M):
    """
    Compute the trace of the matrice

    Var :
        M : matrice

    Output : Tensor (batch_size)
    """
    return M.diagonal(dim1=1, dim2=2).sum(dim=-1)


def trace_equalize(A_in: torch.Tensor, B_out: torch.Tensor):
    """Normlization of the output to have tr(A) back.

        a = tr(A_in)  (trace after BiMap)
        b = tr(B_out) (trace after activation)

    Output : normalized_matricec
    """
    a = trace(A_in) # (batch, subspacedimBiMap, subspacedimBiMap)
    b = trace(B_out) # (batch, subspacedimBiMap, subspacedimBiMap)
    scale = (a / b).view(*a.shape, 1, 1) # (batch, 1, 1) scalaire tr(a)/tr(b)
    normalized_matrice = scale*B_out # Calcul
    return normalized_matrice


# =============================================================================
# Test
# =============================================================================
def test_equalization():
    torch.manual_seed(0)

    # Matrice de covariance
    covs, labels, doms = load_covmats(DATA_DIR, DB_PREFIX, file_id=FILE_ID)
    X = torch.tensor(covs[:32], dtype=DTYPE, device=DEVICE) # (batch, C, C)
    n_chans = X.shape[-1] # C (ici 22)

    bimap = BiMap(in_features=n_chans, out_features=n_chans // 2).to(DTYPE)
    act = coshP(autograd=False).to(DTYPE) # alpha = 0.5

    A = bimap(X) # SPD apres BiMap
    B = act(A) # apres coshP
    B_norm = trace_equalize(A, B) # normalization


    print(50*"=")
    print(f"TEST sur vraie covariance {DB_PREFIX} (batch={X.shape[0]}, C={n_chans})")
    print(50*"=")
    print(f"trace apres BiMap (A) : {trace(A).mean()} 'EigenValue max of A : {torch.linalg.eigvalsh(A).max().item()}")
    print(f"trace apres Activation (B) : {trace(B).mean()} 'EigenValue max of A : {torch.linalg.eigvalsh(B).max().item()}")
    print(f"trace apres normalization (B_norm) : {trace(B_norm).mean()} 'EigenValue max of A : {torch.linalg.eigvalsh(B_norm).max().item()}")
    print(f"  tr(B_eq) == tr(A) ? {torch.allclose(trace(B_norm), trace(A))}")


if __name__ == "__main__":
    test_equalization()