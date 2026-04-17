#!/usr/bin/env python3
"""Test FC forward with actual CIFAR data"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os
import pickle

so = os.path.join(os.path.dirname(__file__), "..", "cpp", "libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)
lib.gpu_malloc.restype = c_void_p
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

BATCH = 64
FC_IN = 2304
OUT_F = 10

# SAME fc_w init as test_fc_alone.py
np.random.seed(42)
fc_w = np.random.randn(OUT_F * FC_IN).astype(np.float32) * 0.05
fc_b = np.zeros(OUT_F, dtype=np.float32)

print(f"fc_w: min={fc_w.min():.4f} max={fc_w.max():.4f}")

dfw = lib.gpu_malloc(OUT_F * FC_IN * 4)
dfb = lib.gpu_malloc(OUT_F * 4)
dfi = lib.gpu_malloc(BATCH * FC_IN * 4)
dfo = lib.gpu_malloc(BATCH * OUT_F * 4)

lib.gpu_memcpy_h2d(dfw, fc_w.ctypes.data, OUT_F * FC_IN * 4)
lib.gpu_memcpy_h2d(dfb, fc_b.ctypes.data, OUT_F * 4)

# Load actual CIFAR data
data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:BATCH].astype(np.float32) / 255.0
    x = x.reshape(-1, 3, 32, 32)

# Create fake h_pool2 from CIFAR-like data (reshaped to FC_IN)
fake_pool2 = x.reshape(BATCH, -1)[:, :FC_IN]
print(f"fake_pool2 (from CIFAR x): min={fake_pool2.min():.4f} max={fake_pool2.max():.4f}")

lib.gpu_memcpy_h2d(dfi, fake_pool2.ctypes.data, BATCH * FC_IN * 4)
lib.dense_forward(dfi, dfw, dfb, dfo, BATCH, FC_IN, OUT_F)
h_out = np.zeros(BATCH * OUT_F, dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, dfo, BATCH * OUT_F * 4)
print(f"FC with fake_pool2: min={h_out.min():.4f} max={h_out.max():.4f}")

lib.gpu_free(dfw); lib.gpu_free(dfb); lib.gpu_free(dfi); lib.gpu_free(dfo)
print("DONE")
