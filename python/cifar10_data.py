"""CIFAR-10 loading and preprocessing helpers."""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIRNAME = "cifar-10-batches-py"
REQUIRED_FILES = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
)
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def normalize_cifar(x):
    return ((x - CIFAR_MEAN) / CIFAR_STD).astype(np.float32)


def cifar10_ready(data_root):
    root = Path(data_root)
    return all((root / name).exists() for name in REQUIRED_FILES)


def _safe_extract(tar, path):
    base = Path(path).resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if base not in target.parents and target != base:
            raise RuntimeError(f"Unsafe path in CIFAR-10 archive: {member.name}")
    tar.extractall(path)


def prepare_cifar10(data_root, download=True):
    data_root = Path(data_root)
    if cifar10_ready(data_root):
        return data_root

    if not download:
        missing = [name for name in REQUIRED_FILES if not (data_root / name).exists()]
        raise FileNotFoundError(
            "CIFAR-10 Python batch files are missing:\n"
            f"  data_root={data_root}\n"
            f"  missing={missing}\n\n"
            "Prepare the dataset with:\n"
            "  python3 python/prepare_cifar10.py\n"
            "or manually place the extracted cifar-10-batches-py directory under data/."
        )

    data_parent = data_root.parent
    data_parent.mkdir(parents=True, exist_ok=True)
    archive_path = data_parent / CIFAR10_ARCHIVE
    if not archive_path.exists():
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract(tar, data_parent)

    extracted_root = data_parent / CIFAR10_DIRNAME
    if extracted_root != data_root and extracted_root.exists() and not data_root.exists():
        extracted_root.rename(data_root)

    if not cifar10_ready(data_root):
        raise RuntimeError(f"CIFAR-10 preparation finished but required files are still missing in {data_root}")
    return data_root


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


def load_cifar10(data_root, n_train=8000, n_val=2000, seed=None, train_batch_ids=(1,), download=True):
    prepare_cifar10(data_root, download=download)
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
