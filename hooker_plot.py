import torch
from collections import defaultdict
import math

from test_layer_equalization import trace
# =====================================================================================


class LayerPlot:
    def __init__(self):
        # Every stats for every batch in each epoch
        self.batch_stats = defaultdict(lambda: defaultdict(list)) # we can do batch_stats[layer_name]['mean_eig'].append(mean_eig)
        # Final stats for each epoch
        self.epochs_stats = defaultdict(lambda: defaultdict(list)) # epochs_stats[layer_name]['max_eig']

    # =====================================================================================
    def hooker(self, model):
        for domain, block in model.domains_block.items(): # Récupère le block spd par domaine
            for layer_name, layer in block.items(): # Récupère le nom de la couche et la couche
                    def forward_hook(module, input, output, name = layer_name): # name créer la key du dico et on définit dedans sinon on peut pas avoir cette info pour register_forward_hook
                        """
                        To know the output of each layer of my model.
                        """
                        try:
                            eig = torch.linalg.eigvalsh(output)
                            eig_mean = eig.mean().item()
                            eig_max = eig.max().item()
                        except Exception: # la matrice a explosé : eigvalsh ne converge plus
                            eig_mean = float("nan")
                            eig_max = float("inf")
                        tr = trace(output)
                        tr_max = tr.max().item()
                        
                        if math.isnan(tr_max) or math.isinf(tr_max) or tr_max > 10000:
                            print("\n +++++++++++++++++ ALERTE trace a EXPLOSE +++++++++++++++++")

                        self.batch_stats[name]['mean_eig'].append(eig_mean)
                        self.batch_stats[name]['max_eig'].append(eig_max)
                        self.batch_stats[name]['max_trace'].append(tr_max)

                    layer.register_forward_hook(forward_hook)
    def compute_epoch_stats(self):
        """
        À appeler à la fin de chaque époque. 
        Fait la moyenne des batchs et vide la mémoire pour l'époque suivante.
        """
        for layer_name, metrics in self.batch_stats.items():
            for metric_name, values_list in metrics.items():
                # Calcul de la moyenne sur tous les batchs de l'époque
                epoch_mean = torch.tensor(values_list).mean().item()
                # Ajout dans l'historique des époques
                self.epochs_stats[layer_name][metric_name].append(epoch_mean)
        
        # On vide
        self.batch_stats.clear()