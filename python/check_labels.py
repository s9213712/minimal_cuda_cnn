#!/usr/bin/env python3
import pickle
with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    print("Keys:", list(batch.keys()))
    print("Labels sample:", batch[b"labels"][:10])
    print("Data shape:", batch[b"data"].shape)
    print("Len labels:", len(batch[b"labels"]))
