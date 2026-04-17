#!/usr/bin/env python3
"""Test FC + Softmax for NaN"""
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

# Load real data
data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    imgs = batch[b"data"][:64].astype(np.float32) / 255.0
    imgs = imgs.reshape(-1, 3, 32, 32)
    labels = np.array(batch[b"labels"][:64])
print(f"Input: {imgs.shape}")

# FC input: reshape as if it's pool2 output (64*6*6=2304)
FC_IN = 64 * 6 * 6
BATCH = 64

# FC weights - use SMALL init
fc_w = np.random.randn(10, FC_IN).astype(np.float32) * 0.01  # Much smaller!
fc_b = np.zeros(10, dtype=np.float32)

d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(40)
d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)

# Use real CIFAR images but pretend they're FC input
# Actually just use random FC input
h_fc_in = np.random.randn(BATCH, FC_IN).astype(np.float32) * 0.05

lib.gpu_memcpy_h2d(d_fc_in, h_fc_in.ctypes.data, BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 40)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
print(f"FC out: min={h_out.min():.4f}, max={h_out.max():.4f}")

h_sh = h_out - h_out.max(axis=1, keepdims=True)
exp_o = np.exp(h_sh)
print(f"Exp: min={exp_o.min():.4f}, max={exp_o.max():.4f}")
if np.isnan(exp_o).any() or np.isinf(exp_o).any():
    print("WARNING: exp has inf/nan!")
probs = exp_o / exp_o.sum(axis=1, keepdims=True)
print(f"Probs: min={probs.min():.4f}, max={probs.max():.4f}")
loss = -np.mean(np.log(probs[np.arange(BATCH), labels] + 1e-10))
print(f"Loss: {loss:.4f}")

lib.gpu_free(d_fc_in)
lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
lib.gpu_free(d_fc_out)
