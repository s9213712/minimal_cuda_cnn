#!/usr/bin/env python3
"""Test FC SGD update in isolation"""
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
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

BATCH = 64
FC_IN = 2304
LR = 0.001

np.random.seed(42)
fc_w_orig = np.random.randn(10 * FC_IN).astype(np.float32) * 0.05
fc_b = np.zeros(10, dtype=np.float32)
h_pool2 = np.random.randn(BATCH, FC_IN).astype(np.float32) * 0.05
y = np.array([6, 9, 9, 4, 1, 1, 2, 7] + [3] * 56)  # First 8 labels

dfw = lib.gpu_malloc(10 * FC_IN * 4)
dfb = lib.gpu_malloc(40)
dfi = lib.gpu_malloc(BATCH * FC_IN * 4)
dfo = lib.gpu_malloc(BATCH * 10 * 4)

lib.gpu_memcpy_h2d(dfw, fc_w_orig.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(dfb, fc_b.ctypes.data, 40)
lib.gpu_memcpy_h2d(dfi, h_pool2.ctypes.data, BATCH * FC_IN * 4)

# Forward
lib.dense_forward(dfi, dfw, dfb, dfo, BATCH, FC_IN, 10)
h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, dfo, BATCH * 10 * 4)
print(f"Forward: h_out min={h_out.min():.4f}, max={h_out.max():.4f}")

# Compute loss
h_sh = h_out - h_out.max(axis=1, keepdims=True)
exp_o = np.exp(h_sh)
probs = exp_o / exp_o.sum(axis=1, keepdims=True)
loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
print(f"Loss: {loss:.4f}")

# Compute gradient
labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot
grad_fc_w = d_loss.T @ h_pool2
grad_fc_w_c = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
print(f"Grad: min={grad_fc_w_c.min():.4f}, max={grad_fc_w_c.max():.4f}")

# Check weight before update
h_fw_before = np.zeros(10 * FC_IN, dtype=np.float32)
lib.gpu_memcpy_d2h(h_fw_before.ctypes.data, dfw, 10 * FC_IN * 4)
print(f"Weight before: min={h_fw_before.min():.6f}, max={h_fw_before.max():.6f}")

# SGD update
d_grad = lib.gpu_malloc(10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_grad, grad_fc_w_c.ctypes.data, 10 * FC_IN * 4)
print(f"d_grad ptr: {d_grad}")
lib.apply_sgd_update(dfw, d_grad, c_float(LR), 10 * FC_IN)
print("SGD update called")

# Check weight after update
h_fw_after = np.zeros(10 * FC_IN, dtype=np.float32)
lib.gpu_memcpy_d2h(h_fw_after.ctypes.data, dfw, 10 * FC_IN * 4)
print(f"Weight after: min={h_fw_after.min():.6f}, max={h_fw_after.max():.6f}")
print(f"NaN in weight after: {np.isnan(h_fw_after).any()}")
print(f"Inf in weight after: {np.isinf(h_fw_after).any()}")

# Verify manually
expected = h_fw_before - LR * grad_fc_w_c
print(f"Expected: min={expected.min():.6f}, max={expected.max():.6f}")
print(f"Actual matches expected: {np.allclose(h_fw_after, expected, rtol=1e-4)}")

lib.gpu_free(dfw)
lib.gpu_free(dfb)
lib.gpu_free(dfi)
lib.gpu_free(dfo)
lib.gpu_free(d_grad)
print("Done")
