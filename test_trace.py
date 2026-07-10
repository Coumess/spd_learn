import sys
import os
import random
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

sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn/activation_test/utils")
sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn/activation_test/models_")

from preprocessing.data_scripts.get_eeg_data import DomainBatchSampler
from activation_test.models_.model_SPD import modelSPDNet
from hooker_plot import LayerPlot


# =====================================================================
# PLOT : 3 panneaux (mean eig / max eig / max trace) par couche
# Couleur des points = époque (dégradé), échelle symlog car ça explose
# =====================================================================
def plot_layer_stats(epochs_stats, activation_name, num_epochs, save_path=None):
    metrics = [("mean_eig", "Mean eigenvalue"),      # clé interne -> titre
               ("max_eig", "Max eigenvalue"),
               ("max_trace", "Max trace")]

    layers = list(epochs_stats.keys())               # bimap1, activation1, ...
    x = np.arange(len(layers))                       # position des couches sur X
    colors = cm.viridis(np.linspace(0, 1, num_epochs))  # dégradé = temps (époque)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    for ax, (key, nice) in zip(axes, metrics):
        for e in range(num_epochs):
            try:
                y = [epochs_stats[l][key][e] for l in layers]   # profil des couches à l'époque e
            except IndexError:
                break                                            # early stopping avant num_epochs
            ax.plot(x, y, color=colors[e], alpha=0.3, lw=1, zorder=1)          # ligne qui relie les couches
            ax.scatter(x, y, color=colors[e], s=45, alpha=0.9,                 # points
                       edgecolors="white", linewidths=0.4, zorder=2)
        ax.set_yscale("symlog")                                  # symlog : gère l'explosion et les valeurs ~0
        ax.set_title(nice, fontweight="bold")
        ax.set_xlabel("SPD layer")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30, ha="right")
        ax.grid(True, which="both", ls="--", alpha=0.3)
    axes[0].set_ylabel("Value (symlog scale)")

    # Colorbar commune = axe temporel (numéro d'époque)
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=num_epochs))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Training epoch", shrink=0.85)

    fig.suptitle(f"Layer diagnostics across training  |  activation: {activation_name}",
                 fontsize=14, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
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

# Une seule Seed et un seul Fold (mode observation)
seed = 1
file = list_files[0]
path_data = os.path.join(path, file)

print(f"\n===== OBSERVATION MODE | SEED {seed} | FICHIER {file} =====")

# Chargement des données
with open(path_data, 'rb') as f:
    data = pickle.load(f)

# Train / Val / Test
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

# On teste les activations l'une après l'autre
for layer_activation in ["reeig", "coshP", "expT"]:
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
    # HOOKER : branche les forward hooks sur chaque couche
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
    for epoch in range(max_epochs):
        spdnet.train()
        train_loss = 0

        for x, y, d in train_loader:
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
            for x, y, d in val_loader:
                x, y, d = x.to(device), y.to(device), d.to(device)
                domain_name = f"domain {d[0].item()}"
                pred = spdnet(x, domain_name)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{max_epochs} | Train loss : {train_loss:.3f} | Val loss : {val_loss:.3f}")

        # Compile les stats de l'époque (moyenne des batchs) pour le plot
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
    # PLOT à la fin de l'entraînement de cette activation
    # =====================================================================
    print(f">>> Création du graphique pour {layer_activation}...")
    epochs_done = epoch + 1   # nb réel d'époques (au cas où early stopping)
    plot_layer_stats(plotter.epochs_stats, layer_activation, epochs_done,
                     save_path=f"trace_plot_{layer_activation}.png")
