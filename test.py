import spd_learn 
import geoopt
import pickle 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

import sys
import os
import torch
import tkinter as t
from tkinter.filedialog import askdirectory, askopenfilename
import random
import copy

sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn")
from preprocessing.data_scripts.get_eeg_data import DomainBatchSampler
from spd_learn.models import TSMNet
from activation_test.models_.model_TSM import TSMNetCustom

# ============================
# Path to the folds of dataset
# ============================
path = t.filedialog.askdirectory(title="Select the folder containing the data")
print("Path of the selecting folder :", path)

list_files= [file for file in os.listdir(path) if file.endswith(".pkl")]

# ===========================
# Fonction set_seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================
# Seed
# ============================
seeds = [1,2,3,4,5]
res_seed = {}
for seed in seeds : 
    print(f"\n ===== SEED {seed} =====")

    res_fold = {}

    for i, file in enumerate(list_files) :
        path_data = os.path.join(path, file)
        print(f"Fichier ; {file}")

        # ---------
        # Load data
        # ---------
        with open (path_data, 'rb') as f:
            data = pickle.load(f)

        # ---------- TEST --------------
        X_test, Y_test, D_test = data['test']
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        D_test = torch.tensor(D_test, dtype=torch.long)

        # ---------- TRAIN --------------
        X_train, Y_train, D_train = data['train']
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.long)
        D_train = torch.tensor(D_train, dtype=torch.long)

        # ---------- VAL --------------
        X_val, Y_val, D_val = data['val']
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.long)
        D_val = torch.tensor(D_val, dtype=torch.long)

        # ------------ Dataloader & Dataset --------------
        batch_size = 32

        # Creates batches with unique domains
        sampler_train = DomainBatchSampler(D_train, batch_size=batch_size, shuffle=True)      
        sampler_test = DomainBatchSampler(D_test, batch_size=batch_size, shuffle=False)
        sampler_val = DomainBatchSampler(D_val, batch_size=batch_size, shuffle=False)

        train_loader = DataLoader(TensorDataset(X_train, Y_train, D_train), batch_sampler=sampler_train)
        val_loader = DataLoader(TensorDataset(X_val, Y_val, D_val), batch_sampler=sampler_val)
        test_loader = DataLoader(TensorDataset(X_test, Y_test, D_test), batch_sampler=sampler_test)

        n_chans = X_train.shape[1]
        n_outputs = len(torch.unique(Y_train))

        res_couche = []

        for layer in ["reeig","coshP","expT"]:
            print(f"\n Training with {layer}")
            set_seed(seed)

        
            #-----------------------
            # Model TSMNet Custom
            #-----------------------
            model = TSMNetCustom(
                activation = layer,
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_temp_filters=8,
                temp_kernel_length=50,
                n_spatiotemp_filters=32,
                n_bimap_filters=16,
                threshold=1e-4
            )

            #----------- Training configuration ------------
            max_epochs = 75
            learning_rate = 0.0005
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            #----------- Loss and Optimization -------------
            criterion = nn.CrossEntropyLoss()
            
            # Adam standard et Adam Riemannian
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=learning_rate)

            #-----------------------
            # Training the model
            #-----------------------
            best_loss = float("inf")
            patience, wait = 10, 0
            best_model_state = None
            
            for epoch in range(max_epochs):
                #--------- TRAINING ----------
                model.train()
                train_loss = 0

                for x, y, d in train_loader:
                    x, y = x.to(device), y.to(device)

                    pred = model(x)
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    train_loss += loss.item()

                #--------- VALIDATION ---------
                model.eval()
                correct, total, val_loss = 0, 0, 0
                all_val_preds, all_val_labels = [], []

                with torch.no_grad():
                    for x, y, d in val_loader:
                        x, y = x.to(device), y.to(device)

                        pred = model(x)
                        loss = criterion(pred, y)
                        val_loss += loss.item()

                        predictions = pred.argmax(dim=1)
                        all_val_preds.append(predictions.cpu())
                        all_val_labels.append(y.cpu())
                        
                        correct += (predictions == y).sum().item()
                        total += y.size(0)

                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                val_accuracy = correct / total

                print(f"Epoch {epoch+1}/{max_epochs} | Train loss : {train_loss:.3f} | Val loss : {val_loss:.3f} | Val accuracy : {val_accuracy:.3f}")
        
                # EARLY STOPPING 
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_model_state = copy.deepcopy(model.state_dict()) # Save the model properly
                else:
                    wait += 1
                    if wait >= patience:
                        model.load_state_dict(best_model_state)
                        break

            #--------- TEST --------- 
            model.eval()
            all_pred, all_label = [], []

            with torch.no_grad():
                for x, y, d in test_loader:
                    x, y = x.to(device), y.to(device)

                    pred_test = model(x)
                    predictions_test = pred_test.argmax(dim=1)

                    all_pred.append(predictions_test.cpu())
                    all_label.append(y.cpu())

            predictions_test = torch.cat(all_pred)
            Y_test_full = torch.cat(all_label)

            # Balanced_accuracy on the test
            test_balanced_accuracy = balanced_accuracy_score(Y_test_full.numpy(), predictions_test.numpy())
            print(f"> Test Balanced Accuracy pour {file} (Seed {seed}) : {test_balanced_accuracy:.4f}")

            # On sauvegarde le résultat pour ce fold
            res_couche.append(test_balanced_accuracy)
    
        # On sauvegarde la liste des 3 scores [reeig, coshP, expT] pour ce fold
        res_fold[i] = res_couche
    
    # On sauvegarde les résultats de tous les folds pour cette seed
    res_seed[seed] = res_fold

#-----------------------------------------------------------
# Converting the results into a vector y = 1D
#-----------------------------------------------------------
y_results = []
for seed in res_seed:
    for fold in res_seed[seed]:
        y_results.append(res_seed[seed][fold])

y_results = np.array(y_results)

# Save .csv 
np.savetxt(r"results_TSMNet_Activations.csv", y_results, delimiter=",", header="reeig, coshP, expT", comments="")

print("\n Finished")