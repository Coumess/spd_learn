"""
Preprocessing pipeline for raw EEG data into k-fold epoch files.

Loads continuous .npz EEG files, epochs them using the metadata in .yml files,
applies optional bandpass filtering, splits per domain into k-folds, and saves
fold_k.pkl files.

Usage:
    python -m data_scripts.preprocess_raw_eeg --config configs/config_raw.yaml

Each fold_k.pkl contains:
    {
        "train": (X, y, d),   # X: (n_trials, n_chans, n_times)
        "val":   (X, y, d),
        "test":  (X, y, d),
    }
"""
# ========================
# IMPORTS
# ========================
import os

from pathlib import Path

#to avoid julia core dumped
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import pickle

from collections import defaultdict

import numpy as np
import yaml
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import KFold

# project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.data_helpers import make_data_signature, write_metadata, load_config



def _load_yml(yml_path: str) -> dict:
    with open(yml_path) as f:
        return yaml.safe_load(f)


def _find_files(data_dir: str, db_prefix: str):
    """Return sorted list of (npz_path, yml_path) for a dataset."""
    pairs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        base = fname[:-4]
        yml_path = os.path.join(data_dir, base + ".yml")
        if not os.path.exists(yml_path):
            continue
        pairs.append((os.path.join(data_dir, fname), yml_path))
    if not pairs:
        raise FileNotFoundError(
            f"No .npz/.yml pairs found in {data_dir}"
        )
    return pairs


# ========================
# Filtering
# ========================

def _bandpass(signal: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """Bandpass filter signal (n_samples, n_chans) using a 4th-order Butterworth."""
    sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal, axis=0).astype(np.float32)


# ========================
# Epoching
# ========================

def _epoch_file(
    npz_path: str,
    yml_path: str,
    l_freq: float | None,
    h_freq: float | None,
    target_labels: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one .npz file and return epochs (n_trials, n_chans, n_times) and labels.

    Epoching uses offset and windowlength from the .yml metadata:
        epoch = signal[onset + offset : onset + offset + windowlength]

    Parameters
    ----------
    target_labels : list[int] or None
        Keep only epochs whose stimulus label is in this list.
        None = keep all non-zero labels.

    Returns
    -------
    X : (n_trials, n_chans, n_times)
    y : (n_trials,)  — 0-indexed class labels
    """
    meta = _load_yml(yml_path)
    data = np.load(npz_path)

    signal = data["data"].astype(np.float32)   # (n_samples, n_chans)
    stim = data["stim"]                         # (n_samples,)

    sfreq = meta["acquisition"]["samplingrate"]
    offset = meta["stim"]["offset"]             # samples
    win = meta["stim"]["windowlength"]          # samples
    label_map = meta["stim"]["labels"]          # e.g. {"left_hand": 1, "right_hand": 2}

    # Optional bandpass filter on continuous signal (cheaper than per-epoch)
    if l_freq is not None and h_freq is not None:
        signal = _bandpass(signal, sfreq, l_freq, h_freq)

    # Detect trial onsets: first sample of each non-zero run
    is_event = stim != 0
    onsets = np.where(np.diff(is_event.astype(int)) == 1)[0] + 1

    # Build reverse map: integer stim value → 0-indexed class index
    if target_labels is not None:
        keep = set(target_labels)
    else:
        keep = set(label_map.values())

    # Sort kept labels so class indices are deterministic
    sorted_labels = sorted(keep)
    int2cls = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

    epochs, labels = [], []
    for onset in onsets:
        ev_val = int(stim[onset])
        if ev_val not in keep:
            continue
        start = onset + offset
        end = start + win
        if end > len(signal):
            continue
        epochs.append(signal[start:end].T)   # (n_chans, n_times)
        labels.append(int2cls[ev_val])

    if not epochs:
        raise ValueError(f"No valid epochs found in {npz_path}")

    X = np.stack(epochs, axis=0)             # (n_trials, n_chans, n_times)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ========================
# Main preprocessing function
# ========================

def preprocess_raw_eeg(
    *,
    db_prefix,
    data_dir,
    processed_root,
    k_folds,
    seed,
    train_pct,
    test_pct,
    l_freq,
    h_freq,
    target_labels,
) -> str:
    """
    Epoch, split per domain, and save k-fold pickle files for raw EEG data.

    Returns the output directory path.
    """
    assert train_pct + test_pct <= 1.0

    # ---------- data signature ----------
    signature = make_data_signature(
        db_prefix=db_prefix,
        data_dir=str(data_dir),
        k_folds=k_folds,
        seed=seed,
        train_pct=train_pct,
        test_pct=test_pct,
        l_freq=l_freq,
        h_freq=h_freq,
        target_labels=sorted(target_labels) if target_labels else None,
    )

    out_dir = os.path.join(processed_root, signature)
    os.makedirs(out_dir, exist_ok=True)

    # Check if already done
    done = all(
        os.path.exists(os.path.join(out_dir, f"fold_{k}.pkl"))
        for k in range(k_folds)
    )
    if done:
        print(f"Already preprocessed — found {k_folds} fold files in {out_dir}")
        return out_dir

    # Load all files
    pairs = _find_files(data_dir, db_prefix)
    print(f"Found {len(pairs)} file(s) for {db_prefix} in {data_dir}")

    # Each file = one domain (subject × session pair)
    all_X, all_y, all_d = [], [], []
    for dom_id, (npz_path, yml_path) in enumerate(pairs):
        print(f"  Epoching domain {dom_id}: {os.path.basename(npz_path)}")
        X_dom, y_dom = _epoch_file(npz_path, yml_path, l_freq, h_freq, target_labels)
        all_X.append(X_dom)
        all_y.append(y_dom)
        all_d.append(np.full(len(y_dom), dom_id, dtype=np.int64))
        print(f"    -> {X_dom.shape[0]} trials, shape {X_dom.shape[1:]}, classes {np.unique(y_dom)}")

    # K-fold split per domain
    rng = np.random.default_rng(seed)
    folds_data = defaultdict(lambda: {"train": [], "val": [], "test": []})
    folds_meta = defaultdict(lambda: {"train": [], "val": [], "test": []})

    for dom_id, (X_dom, y_dom, d_dom) in enumerate(zip(all_X, all_y, all_d)):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(X_dom)):
            n_train = int(train_pct * len(trainval_idx))
            perm = rng.permutation(len(trainval_idx))
            train_idx = trainval_idx[perm[:n_train]]
            val_idx = trainval_idx[perm[n_train:]]

            folds_data[fold_id]["train"].append(
                (X_dom[train_idx], y_dom[train_idx], d_dom[train_idx])
            )
            folds_data[fold_id]["val"].append(
                (X_dom[val_idx], y_dom[val_idx], d_dom[val_idx])
            )
            folds_data[fold_id]["test"].append(
                (X_dom[test_idx], y_dom[test_idx], d_dom[test_idx])
            )

            folds_meta[fold_id]["train"].append(train_idx)
            folds_meta[fold_id]["val"].append(val_idx)
            folds_meta[fold_id]["test"].append(test_idx)

    # Save fold pkls
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
        n_tr = fold_data["train"][0].shape[0]
        n_te = fold_data["test"][0].shape[0]
        print(f"Saved fold {fold_id}: train={n_tr}, test={n_te} -> {pkl_path}")

    # Save metadata
    meta_path = os.path.join(out_dir, "metadata.yaml")
    if not os.path.exists(meta_path):
        write_metadata(
            meta_path,
            data_cfg=dict(
                db_prefix=db_prefix,
                data_dir=str(data_dir),
                k_folds=k_folds,
                seed=seed,
                train_pct=train_pct,
                test_pct=test_pct,
                l_freq=l_freq,
                h_freq=h_freq,
                target_labels=target_labels,
                raw=True,
            ),
            experiment_cfg=dict(signature=signature, output_root=str(processed_root)),
            folds=folds_meta,
        )

    return out_dir


# ========================
# main
# ========================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw EEG data into k-fold epoch files for TSMNet."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_cfg = cfg["data"]["raw_eeg"]
    exp_cfg = cfg["experiment"]

    preprocess_raw_eeg(
        db_prefix=raw_cfg["db_prefix"],
        data_dir=raw_cfg["data_dir"],
        processed_root=raw_cfg["processed_root"],
        k_folds=raw_cfg.get("k_folds", 5),
        seed=exp_cfg["seed"],
        train_pct=raw_cfg.get("train_pct", 0.7),
        test_pct=raw_cfg.get("test_pct", 0.15),
        l_freq=raw_cfg.get("l_freq", 4.0),
        h_freq=raw_cfg.get("h_freq", 40.0),
        target_labels=raw_cfg.get("target_labels", None),
    )


if __name__ == "__main__":
    main()
