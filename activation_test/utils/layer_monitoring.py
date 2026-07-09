import torch
import torch.nn.functional as F

from collections import defaultdict

import matplotlib.pyplot as plt

r"""
=============================================================================
LayerMonitor : suivi des matrices a chaque couche, a chaque epoch.

Pour un modelSPDNet (structure model.domains_block[domain][layer_name]), on
enregistre a CHAQUE forward, pour la sortie SPD de chaque couche :

    - eigenvalue mean   (moyenne des valeurs propres)
    - eigenvalue max    (plus grande valeur propre)
    - trace max         (plus grande trace du batch)

et, une fois par epoch, l'alpha effectif ( softplus(alpha) ) de chaque
couche d'activation (coshP / expT / sinhP / expP...).

Les couches spd_learn etant DEPENDANTES du domaine (un bloc par domaine),
on AGREGE par nom de couche (bimap1, activation1, ...) a travers les
domaines pour garder des courbes lisibles.

Usage (dans test_stat.py) :
-----------------------------------------------------------------------------
    from activation_test.utils.layer_monitoring import LayerMonitor

    spdnet = modelSPDNet(activation=layer, ...)
    monitor = LayerMonitor().attach(spdnet)          # 1) apres creation du modele

    for epoch in range(max_epochs):
        # ... boucle d'entrainement + validation ...
        monitor.epoch_end()                          # 2) fin de chaque epoch

    monitor.plot(save_path=f"monitor_{layer}.png")   # 3) apres l'entrainement
    monitor.remove()                                 # 4) enleve les hooks
=============================================================================
"""


class LayerMonitor:
    def __init__(self):
        self.hooks = []
        self.layers = {}                       # (domain, layer_name) -> module
        self.records = defaultdict(list)       # (domain, layer_name) -> [(eig_mean, eig_max, trace_max), ...] pour l'epoch en cours

        # historiques agreges par nom de couche, une valeur par epoch
        self.history = defaultdict(lambda: {"eig_mean": [], "eig_max": [], "trace_max": []})
        self.alpha_history = defaultdict(list)  # layer_name -> [alpha_effectif par epoch]

    # -------------------------------------------------------------------------
    # Hook : appele a chaque forward de la couche
    # -------------------------------------------------------------------------
    def _hook(self, domain, layer_name):
        def fn(module, inputs, output):
            with torch.no_grad():
                try:
                    eig = torch.linalg.eigvalsh(output)              # (batch, n)
                    eig_mean = eig.mean().item()
                    eig_max = eig.max().item()
                except Exception:
                    eig_mean = float("nan")
                    eig_max = float("inf")
                tr = output.diagonal(dim1=-2, dim2=-1).sum(dim=-1)   # (batch,)
                trace_max = tr.max().item()
            self.records[(domain, layer_name)].append((eig_mean, eig_max, trace_max))
        return fn

    # -------------------------------------------------------------------------
    # Attache un hook sur chaque couche des blocs de domaine
    # -------------------------------------------------------------------------
    def attach(self, model):
        for domain, block in model.domains_block.items():
            for layer_name, layer in block.items():
                self.layers[(domain, layer_name)] = layer
                h = layer.register_forward_hook(self._hook(domain, layer_name))
                self.hooks.append(h)
        return self

    # -------------------------------------------------------------------------
    # Fin d'epoch : agrege les batches -> une valeur par couche, + alpha
    # -------------------------------------------------------------------------
    def epoch_end(self):
        # noms de couche distincts (bimap1, activation1, ...)
        layer_names = sorted({ln for (_, ln) in self.layers})

        for ln in layer_names:
            means, maxes, tmaxes = [], [], []
            for (dom, l), recs in self.records.items():
                if l != ln or len(recs) == 0:
                    continue
                arr = torch.tensor(recs)                  # (n_batches, 3)
                means.append(arr[:, 0].mean().item())     # moyenne des eig sur les batches
                maxes.append(arr[:, 1].max().item())      # max des eig
                tmaxes.append(arr[:, 2].max().item())     # max des traces
            if means:  # au moins un domaine a produit des donnees
                self.history[ln]["eig_mean"].append(sum(means) / len(means))
                self.history[ln]["eig_max"].append(max(maxes))
                self.history[ln]["trace_max"].append(max(tmaxes))

        # alpha effectif de chaque activation (moyenne sur les domaines)
        for ln in layer_names:
            alphas = []
            for (dom, l), layer in self.layers.items():
                if l == ln and hasattr(layer, "alpha"):
                    alphas.append(F.softplus(layer.alpha).detach().cpu().item())
            if alphas:
                self.alpha_history[ln].append(sum(alphas) / len(alphas))

        self.records = defaultdict(list)   # reset pour l'epoch suivante

    # -------------------------------------------------------------------------
    # Plot : eig_mean, eig_max, trace_max (par couche) + alpha
    # -------------------------------------------------------------------------
    def plot(self, save_path=None, show=False):
        has_alpha = len(self.alpha_history) > 0
        n_panels = 4 if has_alpha else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

        # (1) eigenvalue mean
        ax = axes[0]
        for ln, h in self.history.items():
            ax.plot(range(1, len(h["eig_mean"]) + 1), h["eig_mean"], marker=".", label=ln)
        ax.set_title("Eigenvalue mean")
        ax.set_xlabel("epoch"); ax.set_ylabel("mean eig"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        # (2) eigenvalue max (echelle log : peut exploser)
        ax = axes[1]
        for ln, h in self.history.items():
            ax.plot(range(1, len(h["eig_max"]) + 1), h["eig_max"], marker=".", label=ln)
        ax.set_title("Eigenvalue max"); ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("max eig"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        # (3) trace max (echelle log)
        ax = axes[2]
        for ln, h in self.history.items():
            ax.plot(range(1, len(h["trace_max"]) + 1), h["trace_max"], marker=".", label=ln)
        ax.set_title("Trace max"); ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("max trace"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        # (4) alpha effectif ( softplus(alpha) )
        if has_alpha:
            ax = axes[3]
            for ln, vals in self.alpha_history.items():
                ax.plot(range(1, len(vals) + 1), vals, marker=".", label=ln)
            ax.set_title("alpha effectif (softplus)")
            ax.set_xlabel("epoch"); ax.set_ylabel("alpha"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
            print(f"[LayerMonitor] figure sauvegardee : {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
