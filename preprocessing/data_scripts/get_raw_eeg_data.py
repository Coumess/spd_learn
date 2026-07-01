import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.data_helpers import make_data_signature
from data_scripts.get_eeg_data import DomainBatchSampler


class RawEEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, d: np.ndarray):
        self.X = X      # (n_trials, n_chans, n_times)
        self.y = y
        self.d = d

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "data": torch.from_numpy(self.X[idx]).float(),
            "label": torch.tensor(self.y[idx]).long(),
            "domain": torch.tensor(self.d[idx]).long(),
        }


def get_raw_eeg_loaders(
    *,
    processed_root: str,
    db_prefix: str,
    data_dir: str,
    fold_id: int,
    k_folds: int,
    seed: int,
    train_pct: float = 0.7,
    test_pct: float = 0.15,
    l_freq: float | None = 4.0,
    h_freq: float | None = 40.0,
    target_labels: list[int] | None = None,
    batch_size: int = 32,
    batch_single_dom: bool = True,
    min_batch_size: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load a preprocessed fold and return (train_loader, test_loader, val_loader).

    The signature must match exactly what was used in preprocess_raw_eeg.py.
    """
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
        raw=True,
    )

    fold_path = os.path.join(processed_root, signature, f"fold_{fold_id}.pkl")
    if not os.path.exists(fold_path):
        raise FileNotFoundError(
            f"Preprocessed fold not found: {fold_path}\n"
            "Run preprocess_raw_eeg first:\n"
            "  python -m data_scripts.preprocess_raw_eeg --config configs/config_raw.yaml"
        )

    with open(fold_path, "rb") as f:
        data = pickle.load(f)

    def build(split):
        X, y, d = data[split]
        return RawEEGDataset(X, y, d), d

    train_set, doms_train = build("train")
    val_set, doms_val = build("val")
    test_set, doms_test = build("test")

    if batch_single_dom:
        def make_sampler(dom_array, shuffle):
            return DomainBatchSampler(
                dom_array,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                min_batch_size=min_batch_size,
            )
        train_loader = DataLoader(train_set, batch_sampler=make_sampler(doms_train, True))
        val_loader = DataLoader(val_set, batch_sampler=make_sampler(doms_val, False))
        test_loader = DataLoader(test_set, batch_sampler=make_sampler(doms_test, False))
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
