#!/usr/bin/env python3
"""Debug FC gradient flow"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import pickle

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

BATCH, FC_IN, OUT = 4, 8, 3

np.random.seed(42)
fc_w = np.random.randn(OUT * FC_IN).astype(np.float32) * 0.1
fc_b = np.zeros(OUT, dtype=np.float32)
x = np.random.randn(BATCH, FC_IN).astype(np.float32) * 0.1
y = np.array([2, 0, 1, 0])  # one sample per class

print("Initial fc_w:", fc_w.reshape(OUT, FC_IN))

d_fc_w = lib.gpu_malloc(OUT * FC_IN * 4)
d_fc_b = lib.gpu_malloc(OUT * 4)
d_x = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, OUT * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, OUT * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * FC_IN * 4)

d_out = lib.gpu_malloc(BATCH * OUT * 4)
lib.dense_forward(d_x, d_fc_w, d_fc_b, d_out, BATCH, FC_IN, OUT)

h_out = np.zeros((BATCH, OUT), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_out, BATCH * OUT * 4)
print("\nFC output:")
print(h_out)

# Softmax
h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
exp_out = np.exp(h_out_shifted)
probs = exp_out / exp_out.sum(axis=1, keepdims=True)
print("\nProbs:")
print(probs)
print("True labels:", y)

# Loss
loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
print(f"\nLoss: {loss:.4f}")

# Backward
labels_onehot = np.zeros((BATCH, OUT), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot
print("\nd_loss (grad from cross-entropy):")
print(d_loss)

# FC weight gradient: grad_fc_w = d_loss.T @ x
grad_fc_w = d_loss.T @ x
print("\nFC weight gradient (Python):")
print(grad_fc_w)
print("Grad stats:", "min", grad_fc_w.min(), "max", grad_fc_w.max())

# Apply SGD update
LR = 0.1
grad_fc_w_clipped = np.clip(grad_fc_w.flatten(), -1.0, 1.0).astype(np.float32)
d_fc_grad_w = lib.gpu_malloc(OUT * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_grad_w, grad_fc_w_clipped.ctypes.data, OUT * FC_IN * 4)
lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR), OUT * FC_IN)

# Read back updated weights
lib.gpu_memcpy_d2h(fc_w.ctypes.data, d_fc_w, OUT * FC_IN * 4)
print("\nUpdated fc_w:")
print(fc_w.reshape(OUT, FC_IN))

# Forward again
lib.dense_forward(d_x, d_fc_w, d_fc_b, d_out, BATCH, FC_IN, OUT)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_out, BATCH * OUT * 4)
print("\nFC output after update:")
print(h_out)

exp_out2 = np.exp(h_out - h_out.max(axis=1, keepdims=True))
probs2 = exp_out2 / exp_out2.sum(axis=1, keepdims=True)
print("\nProbs after update:")
print(probs2)
loss2 = -np.mean(np.log(probs2[np.arange(BATCH), y] + 1e-10))
print(f"Loss after update: {loss2:.4f}")

lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
lib.gpu_free(d_x)
lib.gpu_free(d_out)
lib.gpu_free(d_fc_grad_w)
print("\nDone!")
