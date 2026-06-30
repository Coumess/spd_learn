"""
Statistical comparison of ReEig / coshP / expT on TSMNet.

Loops over 5 seeds x 5 folds, trains TSMNet with each activation,
and saves balanced accuracy results to CSV.

Data: raw EEG epochs produced by preprocess_raw_eeg.py
      (shape: n_trials x n_chans x n_times)
"""

import sys
import os
import copy
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import geoopt
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, r"D:/BCI/spd_learn")
sys.path.insert(0, r"D:/BCI/spd_learn/activation_test")

from preprocessing.data_scripts.get_eeg_data import DomainBatchSampler
from activation_test.models_.model_TSM import TSMNetCustom


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_DIR = r"D:/BCI/processed_data_raw"   # folder produced by preprocess_raw_eeg.py
RESULTS_DIR = r"D:/BCI/spd_learn/activation_test/results_TSMNet"
RESULTS_FILE = "results_TSMNet_BNCI2014001_LR.csv"

ACTIVATIONS = ["reeig", "coshP", "expT"]
SEEDS = [1, 2, 3, 4, 5]
MAX_EPOCHS = 75
LR = 0.005
BATCH_SIZE = 32
PATIENCE = 10


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_fold_dir(data_dir: str) -> str:
    """Find the preprocessed fold directory (single hash subfolder)."""
    subdirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No preprocessed data found in {data_dir}")
    if len(subdirs) > 1:
        # Take the most recently modified
        subdirs.sort(key=os.path.getmtime, reverse=True)
        print(f"[WARNING] Multiple preprocessed directories found, using: {subdirs[0]}")
    return subdirs[0]


def load_fold(fold_path: str, batch_size: int):
    """Load one fold .pkl and return (train_loader, val_loader, test_loader, n_chans, n_outputs)."""
    with open(fold_path, "rb") as f:
        data = pickle.load(f)

    def to_tensors(split):
        X, y, d = data[split]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(d, dtype=torch.long),
        )

    X_train, Y_train, D_train = to_tensors("train")
    X_val,   Y_val,   D_val   = to_tensors("val")
    X_test,  Y_test,  D_test  = to_tensors("test")

    sampler_train = DomainBatchSampler(D_train, batch_size=batch_size, shuffle=True)
    sampler_val   = DomainBatchSampler(D_val,   batch_size=batch_size, shuffle=False)
    sampler_test  = DomainBatchSampler(D_test,  batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(TensorDataset(X_train, Y_train, D_train), batch_sampler=sampler_train)
    val_loader   = DataLoader(TensorDataset(X_val,   Y_val,   D_val),   batch_sampler=sampler_val)
    test_loader  = DataLoader(TensorDataset(X_test,  Y_test,  D_test),  batch_sampler=sampler_test)

    n_chans   = X_train.shape[1]
    n_outputs = len(torch.unique(Y_train))

    return train_loader, val_loader, test_loader, n_chans, n_outputs


def train_and_eval(model, train_loader, val_loader, test_loader, device):
    """Train with early stopping, return test balanced accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=LR)

    best_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y, d in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- val ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y, d in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                all_preds.append(pred.argmax(dim=1).cpu())
                all_labels.append(y.cpu())
        val_loss /= len(val_loader)
        val_bacc = balanced_accuracy_score(
            torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(),
        )
        print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS} | train={train_loss:.3f} | val={val_loss:.3f} | val_bacc={val_bacc:.3f}")

        # --- early stopping ---
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)

    # --- test ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, d in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)

    bacc = balanced_accuracy_score(
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )
    return bacc


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

fold_dir = find_fold_dir(DATA_DIR)
fold_files = sorted([f for f in os.listdir(fold_dir) if f.endswith(".pkl")])
print(f"Found {len(fold_files)} fold(s) in {fold_dir}\n")

res_seed = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    res_fold = {}

    for fold_i, fname in enumerate(fold_files):
        print(f"\n--- Fold {fold_i} ({fname}) ---")
        fold_path = os.path.join(fold_dir, fname)
        train_loader, val_loader, test_loader, n_chans, n_outputs = load_fold(fold_path, BATCH_SIZE)
        print(f"n_chans={n_chans}, n_outputs={n_outputs}")

        res_couche = []
        for activation in ACTIVATIONS:
            print(f"\n  [activation={activation}]")
            set_seed(seed)

            model = TSMNetCustom(
                activation=activation,
                n_chans=n_chans,
                n_outputs=n_outputs,
            ).to(device)

            bacc = train_and_eval(model, train_loader, val_loader, test_loader, device)
            print(f"  => Test balanced accuracy: {bacc:.4f}")
            res_couche.append(bacc)

        res_fold[fold_i] = res_couche
    res_seed[seed] = res_fold


# ─────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────
y = []
for seed in res_seed:
    for fold in res_seed[seed]:
        y.append(res_seed[seed][fold])

y = np.array(y)  # shape: (n_seeds * n_folds, n_activations)

csv_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
np.savetxt(csv_path, y, delimiter=",", header=",".join(ACTIVATIONS), comments="")
print(f"\nResults saved to {csv_path}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY (mean +/- std over seeds x folds)")
print(f"{'='*60}")
for i, act in enumerate(ACTIVATIONS):
    vals = y[:, i]
    print(f"  {act:10s}: {vals.mean():.4f} +/- {vals.std():.4f}")
