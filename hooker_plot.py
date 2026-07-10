import torch
from collections import defaultdict
import math

from test_layer_equalization import trace
from spd_learn.functional.numerical import get_epsilon
# =====================================================================================


class LayerPlot:
    def __init__(self):
        # Toutes les stats de chaque batch dans une epoch
        self.batch_stats = defaultdict(lambda: defaultdict(list)) # batch_stats[layer_name]['mean_eig'].append(...)
        # Stats finales par epoch
        self.epochs_stats = defaultdict(lambda: defaultdict(list)) # epochs_stats[layer_name]['max_eig']

        self._alerted = set()                                 # couches deja alertees cette epoch (evite le spam)
        self.log_floor = get_epsilon(torch.float32, "eigval_log")  # seuil de clamp de LogEig (float32 ~ 1.19e-5)

    # =====================================================================================
    def hooker(self, model):
        for domain, block in model.domains_block.items(): # Récupère le block spd par domaine
            for layer_name, layer in block.items(): # Récupère le nom de la couche et la couche
                    def forward_hook(module, input, output, name = layer_name): # name créer la key du dico
                        """
                        Récupère les stats spectrales de la sortie SPD de chaque couche.
                        """
                        try:
                            eig = torch.linalg.eigvalsh(output)
                            eig_min = eig.min().item()
                            eig_mean = eig.mean().item()
                            eig_max = eig.max().item()
                            cond = eig_max / eig_min if eig_min > 0 else float("inf") # conditionnement λmax/λmin
                        except Exception: # eigvalsh a divergé : la matrice est foutue
                            eig_min = eig_mean = float("nan")
                            eig_max = cond = float("inf")
                        tr_max = trace(output).max().item()

                        # Alerte (1 fois / couche / epoch) : dit QUELLE couche et POURQUOI c'est risqué
                        danger = (not math.isfinite(cond)) or cond > 1e6 or eig_min < self.log_floor
                        if danger and name not in self._alerted:
                            self._alerted.add(name)
                            print(f"[ALERTE] '{name}' mal conditionne : cond={cond:.2e}, "
                                  f"min_eig={eig_min:.2e}, max_eig={eig_max:.2e}, trace_max={tr_max:.2e} "
                                  f"-> LogEig (eigh/log) risque de planter")

                        self.batch_stats[name]['mean_eig'].append(eig_mean)
                        self.batch_stats[name]['max_eig'].append(eig_max)
                        self.batch_stats[name]['min_eig'].append(eig_min)
                        self.batch_stats[name]['cond'].append(cond)
                        self.batch_stats[name]['max_trace'].append(tr_max)

                    layer.register_forward_hook(forward_hook)

    def compute_epoch_stats(self):
        """
        À appeler à la fin de chaque époque.
        Fait la moyenne des batchs et vide la mémoire pour l'époque suivante.
        """
        for layer_name, metrics in self.batch_stats.items():
            for metric_name, values_list in metrics.items():
                # Moyenne sur tous les batchs de l'époque
                epoch_mean = torch.tensor(values_list).mean().item()
                self.epochs_stats[layer_name][metric_name].append(epoch_mean)

        # On vide pour l'epoch suivante
        self.batch_stats.clear()
        self._alerted.clear()
