import sys
# sys.path.insert(0, r"C:\Users\andrieue\Desktop\PythonPackages")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn")
import spd_learn 
import geoopt
import pickle 
import numpy as np
# sys.path.insert(0, r"C:\Users\andrieue\Desktop\PythonPackages")
import matplotlib.pyplot as plt 
from sklearn.metrics import balanced_accuracy_score
from spd_learn.modules import BiMap, ReEig, LogEig

import os
import random
import copy
import json
import datetime
import subprocess
import tkinter as t
from tkinter.filedialog import askdirectory, askopenfilename

sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn/activation_test/utils")
sys.path.insert(0, r"C:/Users/coumesa/Documents/BCI/spd_learn/activation_test/models_")

from preprocessing.data_scripts.get_eeg_data  import DomainBatchSampler 
from activation_test.models_.model_SPD import modelSPDNet
from hooker_plot import LayerPlot


#-----------------------------------------------
# Reproductibility - def set_seed()
#-----------------------------------------------
def set_seed(seed : int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#-----------------------------------------------
# Extraction des paramètres appris (W des BiMap, alpha des activations)
#-----------------------------------------------
def extract_learned_params(model):
    params = {}
    for domain, block in model.domains_block.items():
        block_params = {}
        for name, layer in block.items():
            entry = {"type": type(layer).__name__}
            if isinstance(layer, BiMap):
                entry["W"] = layer.weight.detach().cpu().numpy()
            if hasattr(layer, "alpha"):
                entry["alpha_raw"] = layer.alpha.detach().cpu().numpy()
                entry["alpha"] = F.softplus(layer.alpha).detach().cpu().numpy()
            if hasattr(layer, "alphaE"):
                entry["alpha_raw"] = layer.alphaE.detach().cpu().numpy()
                entry["alpha"] = F.softplus(layer.alphaE).detach().cpu().numpy()
            if hasattr(layer, "w"):
                entry["w"] = F.softplus(layer.w).detach().cpu().numpy()
            if hasattr(layer, "K"):
                entry["K"] = layer.K
            block_params[name] = entry
        params[domain] = block_params
    return params

#-----------------------------------------------
# Paramètres scalaires des activations (alpha, w) -> suivi à chaque époque
#-----------------------------------------------
def current_alphas(model):
    out = {}
    for domain, block in model.domains_block.items():
        for name, layer in block.items():
            for p in ("alpha", "alphaE", "w"):
                if hasattr(layer, p):
                    out.setdefault(domain, {})[name] = F.softplus(getattr(layer, p)).detach().cpu().numpy()
    return out

#-----------------------------------------------
# CONFIG de l'expérience (tout est sauvegardé dans <out_name>_config.json)
#-----------------------------------------------
CONFIG = {
    "activations": ["reeig", "cosh", "coshP", "expT"],
    "division": "half",          # "half" | "golden"
    "depth": 1,                  # nombre de couches si division = "half" (ignoré si "golden")
    "n_min": 9,                  # dimension minimale de la dernière couche (golden)
    "seeds": [1, 2, 3, 4, 5],
    "batch_size": 32,
    "max_epochs": 75,
    "lr": 0.005,
    "optimizer": "geoopt.optim.RiemannianAdam",
    "patience": 10,
    "grad_clip": 1.0,
    "threshold": 1e-4,
}

#-----------------------------------------------
# Path to the folds of a dataset
#-----------------------------------------------
path = t.filedialog.askdirectory(title="Select the folder containing the data")            # Path to the folder containing the data
print("Path of selected folder : ", path)

list_files = [ file for file in os.listdir(path) if file.endswith(".pkl")]

dataset = os.path.basename(os.path.normpath(path))
arch = "golden" if CONFIG["division"] == "golden" else f"half{CONFIG['depth']}"
out_name = f"results_{dataset}_{arch}"                                              # préfixe de tous les fichiers de sortie

try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    commit = None
with open(f"{out_name}_config.json", "w") as fc:
    json.dump({**CONFIG, "dataset": dataset, "data_path": path, "fold_files": list_files,
               "git_commit": commit, "date": datetime.datetime.now().isoformat(),
               "torch": torch.__version__, "geoopt": geoopt.__version__}, fc, indent=2)


#-----------------------------------------------
# Seed
#-----------------------------------------------
seeds = CONFIG["seeds"]
res_seed = {}
learned_params = {}
for seed in seeds :
    print(f"\n===== SEED {seed} =====")

    res_fold = {}
    for i,file in enumerate(list_files) :
        
        path_data = os.path.join(path, file)
        print(f"Fichier : {file}")

        #-----------------------
        # Load data
        #-----------------------
        with open(path_data, 'rb') as f : 
            data = pickle.load(f)

        # ---------- TEST --------------
        test_data = data['test']
        X_test, Y_test, D_test = test_data
        print(f"Size of test_data X_test {X_test.shape[1]}")
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        D_test = torch.tensor(D_test)

        # ---------- TRAIN --------------
        train_data = data['train']
        X_train, Y_train, D_train = train_data

        unique_domains = np.unique(D_train)
        domains = [f"domain {d}" for d in unique_domains]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.long)
        D_train = torch.tensor(D_train)

        # ---------- VAL --------------
        val_data = data['val']                                                          
        X_val, Y_val, D_val = val_data
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.long)
        D_val = torch.tensor(D_val)

        # ---------- Dataloader & Dataset --------------
        batch_size = CONFIG["batch_size"]

        sampler_train = DomainBatchSampler(D_train, batch_size=batch_size, shuffle=True)      # Creates batches with unique domains 
        sampler_test = DomainBatchSampler(D_test, batch_size=batch_size, shuffle=False)
        sampler_val = DomainBatchSampler(D_val, batch_size=batch_size, shuffle=False)

        train_loader = DataLoader(TensorDataset(X_train, Y_train, D_train), batch_sampler=sampler_train)
        val_loader = DataLoader(TensorDataset(X_val, Y_val, D_val), batch_sampler=sampler_val)
        test_loader = DataLoader(TensorDataset(X_test, Y_test, D_test), batch_sampler=sampler_test)

        res_couche = []

        for layer in CONFIG["activations"]:
            print(f"\n---> Entraînement avec l'activation : {layer}")
            set_seed(seed)
            n_chans = X_train.shape[1]
            n_outputs = len(torch.unique(Y_train))

            #-----------------------
            # Model
            #-----------------------
            spdnet = modelSPDNet(
                activation = layer,
                division = CONFIG["division"],
                depth = CONFIG["depth"],
                n_min = CONFIG["n_min"],
                n_chans = n_chans,
                n_outputs = n_outputs,
                threshold = CONFIG["threshold"],
                domains = domains
            )
            print(f"Dimensions : {spdnet.dims}")
            plotter = LayerPlot()
            plotter.hooker(spdnet)

            #----------- Training configuration ------------
            max_epochs = CONFIG["max_epochs"]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            spdnet.to(device)

            #----------- Loss and Optimization -------------
            criterion = nn.CrossEntropyLoss()
            optimizer = geoopt.optim.RiemannianAdam(spdnet.parameters(), lr = CONFIG["lr"])

            #-----------------------
            # Training the model
            #-----------------------

            best_loss = float("inf")
            patience, wait = CONFIG["patience"], 0
            stop_epoch, best_epoch = max_epochs, 0
            best_model_state = copy.deepcopy(spdnet.state_dict())
            history = [{"epoch": 0, "alpha": current_alphas(spdnet)}]              # epoch 0 = valeurs initiales

            for epoch in range(max_epochs) :
                #--------- TRAINING ----------
                spdnet.train()                                              
                train_loss = 0                                              #  To stock the loss 

                for x,y,d in train_loader :
                    assert torch.all(d == d[0]), "Batch contains multiple domains!"
                    x = x.to(device)
                    y = y.to(device)
                    d = d.to(device)

                    domain_name = f"domain {d[0].item()}"
                    pred = spdnet(x, domain_name)
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(spdnet.parameters(), CONFIG["grad_clip"])
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                plotter.compute_epoch_stats()                               # stats spectrales des batchs d'entraînement de l'époque

                #--------- VALIDATION ---------
                spdnet.eval()
                correct = 0
                total = 0
                all_val_preds = []
                all_val_labels = []
                val_loss = 0

                with torch.no_grad():
                    for x,y,d in val_loader :
                        x = x.to(device)
                        y = y.to(device)
                        d = d.to(device)

                        domain_name = f"domain {d[0].item()}"
                        pred = spdnet(x, domain_name)
                        loss = criterion(pred, y)
                        val_loss += loss.item()

                        predictions = pred.argmax(dim=1)
                        all_val_preds.append(predictions.cpu())
                        all_val_labels.append(y.cpu())
                        correct += (predictions == y).sum().item()
                        total += y.size(0)

                val_loss /= len(val_loader)
                val_accuracy = correct / total
        
                predictions_val = torch.cat(all_val_preds)
                Y_val_full = torch.cat(all_val_labels)
                val_bal_acc = balanced_accuracy_score(Y_val_full.numpy(),predictions_val.numpy())
                plotter.batch_stats.clear()                                 # on ne garde pas les stats des batchs de validation
                print(f"Epoch {epoch+1}/{max_epochs} | Train loss : {train_loss:.3f} | Val loss : {val_loss:.3f} | Val accuracy : {val_accuracy:.3f}")

                history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_accuracy,
                    "val_bacc": val_bal_acc,
                    "alpha": current_alphas(spdnet),
                })

                # EARLY STOPPING
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    wait = 0
                    best_model_state = copy.deepcopy(spdnet.state_dict())   # Save the best model properly
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping!")
                        stop_epoch = epoch + 1
                        break

            spdnet.load_state_dict(best_model_state)                        # meilleur modèle, même sans early stopping
            plotter.batch_stats.clear()                                     # pas de stats pendant le test
            #--------- TEST --------- 
            spdnet.eval()
            correct = 0
            total = 0
            all_pred = []
            all_label = []

            with torch.no_grad() :
                for x,y,d in test_loader : 
                    x = x.to(device)
                    y = y.to(device)
                    d = d.to(device)

                    domain_name = f"domain {d[0].item()}"
                    pred_test = spdnet(x, domain_name)
                    predictions_test = pred_test.argmax(dim=1)

                    correct += (predictions_test == y).sum().item()
                    total += y.size(0)

                    all_pred.append(predictions_test.cpu())
                    all_label.append(y.cpu())

            predictions_test = torch.cat(all_pred)                                                # To concatenate tensors along a dimension 
            Y_test_full = torch.cat(all_label)

            # Balanced_accuracy on the test
            test_balanced_accuracy = balanced_accuracy_score(Y_test_full.numpy(),predictions_test.numpy())
            print(f"\nTest Balanced Accuracy {test_balanced_accuracy:.4f}")

            res_couche.append(test_balanced_accuracy)

            # Sauvegarde des paramètres appris pour ce (seed, fold, activation)
            learned_params.setdefault(seed, {}).setdefault(file, {})[layer] = {
                "test_bacc": test_balanced_accuracy,
                "dims": spdnet.dims,
                "stop_epoch": stop_epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "history": history,                                        # pertes + alpha à chaque époque
                "spectral": {k: dict(v) for k, v in plotter.epochs_stats.items()},  # min/mean/max eig, cond, trace par couche et par époque
                "params": extract_learned_params(spdnet),
            }
            with open(f"{out_name}.pkl", "wb") as fpk:
                pickle.dump(learned_params, fpk)
        res_fold[i] = res_couche
    res_seed[seed] = res_fold

#-----------------------------------------------------------
# Converting the results into a vector y = 1D
#-----------------------------------------------------------
y = []
for seed in res_seed:
    for fold in res_seed[seed]:
        y.append(res_seed[seed][fold])

y = np.array(y)

# Sauvegarder en CSV
np.savetxt(f"{out_name}.csv", y, delimiter=",", header=", ".join(CONFIG["activations"]), comments="")

#-----------------------------------------------------------
# Sauvegarde lisible des paramètres scalaires (alpha) + accuracy
# (les matrices W complètes sont dans learned_params.pkl)
#-----------------------------------------------------------
with open(f"{out_name}.txt", "w") as fsum:
    for s in learned_params:
        for fl in learned_params[s]:
            for act in learned_params[s][fl]:
                rec = learned_params[s][fl][act]
                fsum.write(f"seed={s} | fold={fl} | activation={act} | test_bacc={rec['test_bacc']:.4f}\n")
                for dom, block in rec["params"].items():
                    for lname, e in block.items():
                        if "alpha" in e:
                            fsum.write(f"    {dom} | {lname} ({e['type']}) : alpha={np.ravel(e['alpha'])}\n")
                fsum.write("\n")

print(f"Résultats sauvegardés : {out_name}_config.json, {out_name}.csv, {out_name}.pkl, {out_name}.txt")