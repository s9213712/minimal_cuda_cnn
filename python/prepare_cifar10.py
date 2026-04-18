#!/usr/bin/env python3
"""Download and extract CIFAR-10 Python batches."""

import os

from cifar10_data import prepare_cifar10


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "data", "cifar-10-batches-py")


def main():
    path = prepare_cifar10(DATA_ROOT, download=True)
    print(f"CIFAR-10 ready at {path}")


if __name__ == "__main__":
    main()
