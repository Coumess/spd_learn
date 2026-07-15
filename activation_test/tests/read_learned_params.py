import sys
import pickle
from collections import defaultdict

import numpy as np

# Chemin du .pkl (par defaut celui produit par test_stat.py)
PATH = sys.argv[1] if len(sys.argv) > 1 else "learned_params.pkl"

with open(PATH, "rb") as f:
    data = pickle.load(f)   # data[seed][fold][activation] = {"test_bacc":..., "params":...}


# =====================================================================
# 1) Resume detaille : chaque (seed, fold, activation)
# =====================================================================
print("=" * 70)
print(f"Fichier : {PATH}")
print("=" * 70)
for seed in sorted(data):
    for fold in data[seed]:
        for act in data[seed][fold]:
            rec = data[seed][fold][act]
            print(f"\nseed={seed} | fold={fold} | activation={act} | test_bacc={rec['test_bacc']:.4f}")
            for dom, block in rec["params"].items():
                for lname, e in block.items():
                    if "W" in e:
                        print(f"    {dom} | {lname:<12} ({e['type']}) : W shape={e['W'].shape}")
                    if "alpha" in e:
                        print(f"    {dom} | {lname:<12} ({e['type']}) : alpha={np.ravel(e['alpha'])}")


# =====================================================================
# 2) Agregats par activation (across seeds & folds)
# =====================================================================
print("\n" + "=" * 70)
print("AGREGATS par activation (across seeds & folds)")
print("=" * 70)

bacc = defaultdict(list)
alphas = defaultdict(lambda: defaultdict(list))   # activation -> couche -> [alpha, ...]

for seed in data:
    for fold in data[seed]:
        for act in data[seed][fold]:
            rec = data[seed][fold][act]
            bacc[act].append(rec["test_bacc"])
            for dom, block in rec["params"].items():
                for lname, e in block.items():
                    if "alpha" in e:
                        alphas[act][lname].append(float(np.ravel(e["alpha"])[0]))

for act in bacc:
    arr = np.array(bacc[act])
    print(f"\n{act} : test_bacc = {arr.mean():.4f} +/- {arr.std():.4f}  (n={len(arr)})")
    for lname, vals in alphas[act].items():
        v = np.array(vals)
        print(f"    alpha[{lname}] = {v.mean():.4f} +/- {v.std():.4f}  (n={len(v)})")


# =====================================================================
# 3) Helper : recuperer une matrice W precise
#    ex: W = get_W(seed=1, fold="fold_0.pkl", activation="expT", layer="bimap1")
# =====================================================================
def get_W(seed, fold, activation, layer="bimap1", domain=None):
    block = data[seed][fold][activation]["params"]
    if domain is None:
        domain = next(iter(block))
    return block[domain][layer]["W"]


def get_alpha(seed, fold, activation, layer="activation1", domain=None):
    block = data[seed][fold][activation]["params"]
    if domain is None:
        domain = next(iter(block))
    return block[domain][layer].get("alpha")
