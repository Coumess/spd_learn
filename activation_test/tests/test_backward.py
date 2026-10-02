"""
Vérifie les backward faits main des activations élément par élément (utils_elementwise.py).

gradcheck compare le gradient du backward (dL/dX et dL/dalpha) à une dérivée par
différences finies. Si une formule du backward est fausse, le test échoue.

Lancer depuis la racine du repo :
    python -m pytest activation_test/tests/test_backward.py -v
ou directement :
    python activation_test/tests/test_backward.py
"""
import os
import sys

import torch
from torch.autograd import gradcheck

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from activation_test.activation.utils_elementwise import (
    cosh_parametric,
    cosh_parametric_tracenorm,
    exp_parametric,
    polynomialFunction,
    sinh_parametric,
    truncated_exponential,
)

DTYPE = torch.float64                                   # float64 obligatoire pour gradcheck


def random_spd(batch=2, n=4, seed=0):
    """Matrices SPD aléatoires, entrées d'ordre 1 (comme après BiMap)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(batch, n, n, generator=g, dtype=DTYPE) / n**0.5
    return A @ A.mT + 0.1 * torch.eye(n, dtype=DTYPE)


def check(fn, param):
    """gradcheck sur (X, param), avec X symétrisé comme dans le réseau."""
    X = random_spd().requires_grad_(True)
    p = param.clone().to(DTYPE).requires_grad_(True)
    sym = lambda X: (X + X.mT) / 2
    assert gradcheck(lambda X, p: fn(sym(X), p), (X, p), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_coshP():
    check(cosh_parametric.apply, torch.tensor(0.8))

def test_coshP_tracenorm():
    check(cosh_parametric_tracenorm.apply, torch.tensor(0.8))

def test_sinhP():
    check(sinh_parametric.apply, torch.tensor(0.8))

def test_expP():
    check(exp_parametric.apply, torch.tensor(0.8))

def test_expT():
    check(lambda X, a: truncated_exponential.apply(X, a, 4), torch.tensor(0.8))

def test_polyAct():
    check(polynomialFunction.apply, torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"OK  {name}")
