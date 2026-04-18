"""CIFAR-10 loading and preprocessing helpers."""

from __future__ import annotations

import os
import pickle

import numpy as np


CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def normalize_cifar(x):
    return ((x - CIFAR_MEAN) / CIFAR_STD).astype(np.float32)


def _load_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x = (batch[b"data"].astype(np.float32) / 255.0).reshape(-1, 3, 32, 32)
    y = np.array(batch[b"labels"])
    return x, y


def _load_training_batches(data_root, batch_ids):
    x_parts = []
    y_parts = []
    for i in batch_ids:
        x_batch, y_batch = _load_batch(os.path.join(data_root, f"data_batch_{i}"))
        x_parts.append(x_batch)
        y_parts.append(y_batch)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def load_cifar10(data_root, n_train=8000, n_val=2000, seed=None, train_batch_ids=(1,)):
    print(f"Loading CIFAR-10 training batches: {train_batch_ids}")
    x_train_all, y_train_all = _load_training_batches(data_root, train_batch_ids)
    print(f"Training samples: {x_train_all.shape[0]}")

    x_test, y_test = _load_batch(os.path.join(data_root, "test_batch"))
    print(f"Test samples: {x_test.shape[0]}")

    if n_train + n_val > x_train_all.shape[0]:
        raise ValueError(f"n_train + n_val exceeds available samples: {n_train} + {n_val}")

    rng = np.random.default_rng(seed) if seed is not None else None
    indices = rng.permutation(x_train_all.shape[0]) if rng else np.random.permutation(x_train_all.shape[0])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    return (
        x_train_all[train_idx],
        y_train_all[train_idx],
        x_train_all[val_idx],
        y_train_all[val_idx],
        x_test,
        y_test,
    )
