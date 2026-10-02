"""
Vérifie que modelSPDNet se construit et s'entraîne (forward + backward) pour chaque
architecture du plan et chaque activation étudiée.

    python -m pytest activation_test/tests/test_model.py -v
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from activation_test.models_.model_SPD import compute_dims, modelSPDNet

ACTIVATIONS = ["reeig", "cosh", "coshP", "expT"]
ARCHS = [("half", 1), ("half", 2), ("half", 3), ("golden", None)]


def test_compute_dims():
    assert compute_dims(64, "half", 3) == [64, 32, 16, 8]
    assert compute_dims(128, "golden") == [128, 79, 49, 30, 19, 12]
    assert compute_dims(100, "golden") == [100, 62, 38, 23, 14, 9]
    with pytest.raises(ValueError):
        compute_dims(13, "golden")                          # 13/phi < 9 : aucune couche


@pytest.mark.parametrize("division, depth", ARCHS)
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_forward_backward(activation, division, depth):
    if activation == "cosh" and (division == "golden" or depth >= 3):
        # cosh >= 1 sur toutes les entrées -> la trace explose (cosh(cosh(...))) : inf/NaN dès 3 couches
        pytest.xfail("cosh sans paramètre explose numériquement à partir de 3 couches")
    torch.manual_seed(0)
    n, domains = 64, ["domain 1", "domain 2"]
    model = modelSPDNet(activation=activation, division=division, depth=depth or 1,
                        n_chans=n, n_outputs=2, domains=domains)

    A = torch.randn(8, n, n) / n**0.5                       # covariances d'ordre 1, en float32 comme test_stat
    X = A @ A.mT + 0.1 * torch.eye(n)

    out = model(X, "domain 1")
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()

    out.sum().backward()
    for name, p in model.domains_block["domain 1"].named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for p in model.domains_block["domain 2"].parameters():
        assert p.grad is None                               # l'autre domaine n'est pas touché
