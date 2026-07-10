import sys
import os
import random
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tkinter as t
from tkinter.filedialog import askdirectory

sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn")
import geoopt
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn/activation_test/utils")
sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn/activation_test/models_")

from preprocessing.data_scripts.get_eeg_data import DomainBatchSampler 
from activation_test.models_.model_SPD import modelSPDNet
from hooker_plot import LayerPlot


# =====================================================================
# FONCTION DE PLOT DÉGRADÉ (SCATTER)
# =====================================================================
def plot_trace_scatter(epochs_stats, activation_name, num_epochs):
    """
    Crée un scatter plot où l'axe X = les couches, l'axe Y = la trace max.
    La couleur des points dépend de l'époque (dégradé).
    """
    # Récupérer le nom de toutes les couches (bimap1, activation1, etc.)
    layers = list(epochs_stats.keys())
    
    plt.figure(figsize=(10, 6))
    
    # Création d'un dégradé de couleurs (ex: viridis va du violet au jaune)
    colors = cm.viridis(np.linspace(0, 1, num_epochs))
    
    for epoch in range(num_epochs):
        # On essaie de récupérer les données, sinon on skip si l'early stopping s'est activé plus tôt
        try:
            traces = [epochs_stats[layer]['max_trace'][epoch] for layer in layers]
            # On trace les points pour cette époque (alpha = 0.7 pour la transparence)
            plt.scatter(layers, traces, color=colors[epoch], alpha=0.7, s=50)
        except IndexError:
            break # L'early stopping a arrêté l'entraînement avant num_epochs
            
    # Ajout d'une barre de couleur sur le côté pour lire les époques
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_epochs))
    plt.colorbar(sm, ax=plt.gca(), label="Numéro de l'Époque (Time)")
    
    plt.title(f"Évolution de la Trace Max par Couche ({activation_name})")
    plt.ylabel("Trace Max")
    plt.xlabel("Couches SPD")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Utiliser une échelle logarithmique si les valeurs explosent vraiment fort
    # plt.yscale('log') 
    
    plt.show()

# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

# Sélection du dossier
path = t.filedialog.askdirectory(title="Select the folder containing the data")
print("Path of selected folder : ", path)
list_files = [f for f in os.listdir(path) if f.endswith(".pkl")]

# Fixer une seule Seed et un seul Fold
seed = 1
file = list_files[0]
path_data = os.path.join(path, file)

print(f"\n===== OBSERVARION MODE | SEED {seed} | FICHIER {file} =====")

# Chargement des données
with open(path_data, 'rb') as f : 
    data = pickle.load(f)

# Traitement Train / Val / Test (Identique à ton code)
X_test, Y_test, D_test = data['test']
X_test, Y_test, D_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long), torch.tensor(D_test)

X_train, Y_train, D_train = data['train']
unique_domains = np.unique(D_train)
domains = [f"domain {d}" for d in unique_domains]
X_train, Y_train, D_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long), torch.tensor(D_train)

X_val, Y_val, D_val = data['val']
X_val, Y_val, D_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long), torch.tensor(D_val)

# Dataloaders
batch_size = 32
sampler_train = DomainBatchSampler(D_train, batch_size=batch_size, shuffle=True)
sampler_test = DomainBatchSampler(D_test, batch_size=batch_size, shuffle=False)
sampler_val = DomainBatchSampler(D_val, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(TensorDataset(X_train, Y_train, D_train), batch_sampler=sampler_train)
val_loader = DataLoader(TensorDataset(X_val, Y_val, D_val), batch_sampler=sampler_val)
test_loader = DataLoader(TensorDataset(X_test, Y_test, D_test), batch_sampler=sampler_test)

# On va tester tes activations l'une après l'autre
for layer_activation in ["reeig", "coshP","expT"]: 
    print(f"\n---> Entraînement avec l'activation : {layer_activation}")
    set_seed(seed)
    
    n_chans = X_train.shape[1]
    n_outputs = len(torch.unique(Y_train))
    
    # Instanciation du modèle
    spdnet = modelSPDNet(
        activation=layer_activation,
        n_chans=n_chans,
        n_outputs=n_outputs,
        threshold=1e-4,
        domains=domains
    )
    
    # ---------------------------------------------
    # HOOKER
    # ---------------------------------------------
    plotter = LayerPlot()
    plotter.hooker(spdnet)
    
    # Configuration
    max_epochs = 75
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spdnet.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = geoopt.optim.RiemannianAdam(spdnet.parameters(), lr=0.005)
    
    best_loss = float("inf")
    patience, wait = 10, 0
    
    # Boucle d'entraînement
    for epoch in range(max_epochs) :                                
        spdnet.train()                                              
        train_loss = 0                                              

        for x, y, d in train_loader :
            assert torch.all(d == d[0]), "Batch contains multiple domains!"
            x, y, d = x.to(device), y.to(device), d.to(device)
            domain_name = f"domain {d[0].item()}"
            
            pred = spdnet(x, domain_name)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(spdnet.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation pour l'early stopping
        spdnet.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, d in val_loader :
                x, y, d = x.to(device), y.to(device), d.to(device)
                domain_name = f"domain {d[0].item()}"
                pred = spdnet(x, domain_name)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train loss : {train_loss:.3f} | Val loss : {val_loss:.3f}")

        # Compiler les stats pour le plot
        plotter.compute_epoch_stats()

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping déclenché !")
                break

    # =====================================================================
    # AFFICHAGE DU PLOT À LA FIN DE L'ENTRAÎNEMENT DE CETTE ACTIVATION
    # =====================================================================
    print(f">>> Création du graphique pour {layer_activation}...")
    # On passe le nombre réel d'époques effectuées (au cas où l'early stopping a frappé)
    epochs_done = epoch + 1
    plot_trace_scatter(plotter.epochs_stats, layer_activation, epochs_done)