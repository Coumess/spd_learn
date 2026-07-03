from sklearn.model_selection import KFold
import os
import sys
import argparse
import pickle

from collections import Counter, defaultdict

import numpy as np

from preprocessing.utils.data_helpers import make_data_signature, load_config, write_metadata

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(project_root)

# ===================================================
# Peut-être à mettre dans get_eeg_data
# ===================================================

def load_raw_tensors(data_dir, db_prefix, file_id=None, verbose=True):
    """Load and concatenate all consistent (tensors(signal eeg), label) arrays"""
    if verbose:
        if file_id:
            msg = f"Loading tensors from {db_prefix} database and file ID {file_id}..."
        else:
            msg = f"Loading tensors from {db_prefix} database (all file IDs)..."
        print(msg)

    tensors_all, labels_all, dom_all, shapes = [], [], [], []
    candidates = []
    
    if file_id: # for a single file (sess-subj)
        prefix = f"{db_prefix}_{file_id}"
        tensor_path = os.path.join(data_dir, f"{prefix}_signal_eeg.npy")
        label_path = os.path.join(data_dir, f"{prefix}_labels_eeg.npy")
        if os.path.exists(label_path):
            tensors = np.load(tensor_path)
            candidates.append((f"{prefix}_signal_eeg.npy", tensors.shape))
            shapes.append(tensors.shape[1:])
    else: # get all sessions if file_id is not specified
        for fname in os.listdir(data_dir):
            if fname.startswith(db_prefix) and fname.endswith("_signal_eeg.npy"):
                prefix = fname.replace("_signal_eeg.npy","")
                tensor_path = os.path.join(data_dir, fname)
                label_path = os.path.join(data_dir, f"{prefix}_labels_eeg.npy")
                if os.path.exists(label_path):
                    tensors = np.load(tensor_path)
                candidates.append((f"{prefix}_signal_eeg.npy", tensors.shape))
                shapes.append(tensors.shape[1:])

    if not candidates:
        raise ValueError(f"No matching files for prefix '{db_prefix}' in {data_dir}")

    # Verify if every sessions has the same number of electrodes
    common_shape = Counter(shapes).most_common(1)[0][0]
    mismatched = [f for f, s in candidates if s[1:] != common_shape]
    if mismatched:
        warnings.warn(
            f"Skipping {len(mismatched)} mismatched files: {mismatched}. Most common shape: {common_shape}"
        )

    dom_id = 0
    for fname, shape in candidates:
        if shape[1:] != common_shape:
            continue
        prefix = fname.replace("_signal_eeg.npy", "")
        tensor_path = os.path.join(data_dir, fname)
        label_path = os.path.join(data_dir, f"{prefix}_labels_eeg.npy")

        tensors = np.load(tensor_path)
        labels = np.load(label_path)

        tensors_all.append(tensors)
        labels_all.append(labels)
        dom_all.extend([dom_id] * len(labels))
        dom_id += 1

    tensors_all = np.concatenate(tensors_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    
    # convert to right class idx e.g. [1,2] becomes [0,1]
    labels_all = np.array([int(el) - 1 for el in labels_all])
    dom_all = np.array(dom_all)

    if verbose:
        print(f"Loaded {len(labels_all)} total trials, shape {common_shape} found at {data_dir}")
        
    return tensors_all, labels_all, dom_all
# ===================================================

def preprocess_raw_eeg_data(
    *,
    db_prefix,
    data_dir,
    file_id,
    train_pct,
    test_pct,
    k_folds,
    seed,
    output_root
):
    """
    Generate K-fold preprocessed Raw EEG data and save to disk.


    """
    signature = make_data_signature(
        db_prefix=db_prefix,
        data_dir=data_dir,
        file_id=file_id,
        train_pct = train_pct,
        test_pct = test_pct,
        k_folds=k_folds,
        seed=seed
    )

    out_dir = os.path.join(output_root, signature)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- load raw data ----------
    signals, labels, domains = load_raw_tensors(
        data_dir=data_dir,
        db_prefix=db_prefix,
        file_id=file_id
        )

    rng = np.random.default_rng(seed)

    # ---------- assemble and save fold pkls ----------
    domain_indices = {dom: np.where(domains == dom)[0] for dom in np.unique(domains)}
    
    folds = defaultdict(lambda: {"train": [], "val": [], "test": []})
    folds_data = defaultdict(lambda: {"train": [], "val": [], "test": []})

    for dom, idx in domain_indices.items():
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(idx)):
            idx_trainval = idx[trainval_idx]
            idx_test = idx[test_idx]

            # Séparation train / val
            n_train = int(train_pct * len(idx_trainval))
            perm = rng.permutation(len(idx_trainval))
            train_idx = idx_trainval[perm[:n_train]]
            val_idx = idx_trainval[perm[n_train:]]

            # Sauvegarde des indices (pour les métadonnées)
            folds[fold_id]["train"].append(train_idx)
            folds[fold_id]["val"].append(val_idx)
            folds[fold_id]["test"].append(idx_test)

            # Sauvegarde des vraies données brutes
            folds_data[fold_id]["train"].append((signals[train_idx], labels[train_idx], domains[train_idx]))
            folds_data[fold_id]["val"].append((signals[val_idx], labels[val_idx], domains[val_idx]))
            folds_data[fold_id]["test"].append((signals[idx_test], labels[idx_test], domains[idx_test]))

    for fold_id in range(k_folds):
        fold_data = {}
        for split in ("train", "val", "test"):
            X_list, y_list, d_list = zip(*folds_data[fold_id][split])
            fold_data[split] = (
                np.concatenate(X_list, axis=0),
                np.concatenate(y_list, axis=0),
                np.concatenate(d_list, axis=0),
            )

        pkl_path = os.path.join(out_dir, f"fold_{fold_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(fold_data, f)
        print(f"Fold {fold_id} sauvegardé dans : {pkl_path}")

    # ------------------ write metadata ------------------------
    metadata_path = os.path.join(out_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        write_metadata(
            metadata_path,
            data_cfg=dict(
                db_prefix=db_prefix,
                data_dir=data_dir,
                file_id=file_id,
                precond=False,
                train_pct=train_pct,
                test_pct=test_pct,
                k_folds=k_folds,
                seed=seed,
            ),
            experiment_cfg=dict(signature=signature, output_root=output_root),
            folds=folds,
        )


def main():
    parser = argparse.ArgumentParser(description="Preprocess RAW EEG data (K-fold, domain-aware).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]["eeg"]
    exp_cfg = cfg["experiment"]

    preprocess_raw_eeg_data(
        db_prefix=data_cfg["db_prefix"],
        data_dir=data_cfg["data_dir"],
        file_id=data_cfg.get("file_id"),
        train_pct=data_cfg.get("train_pct", 0.7),
        test_pct=data_cfg.get("test_pct", 0.15),
        k_folds=data_cfg["k_folds"],
        seed=exp_cfg["seed"],
        output_root=data_cfg["processed_root"]
    )

if __name__ == "__main__":
    main()