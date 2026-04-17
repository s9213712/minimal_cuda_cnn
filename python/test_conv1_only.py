#!/usr/bin/env python3
"""Just Conv1 + Pool1 + FC (no Conv2)"""
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
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

BATCH = 64
LR = 0.001
C1_IN = 3; C1_OUT = 32; H = 32; W = 32; KH1 = 3; KW1 = 3
outH1 = 30; outW1 = 30; poolH1 = 15; poolW1 = 15
# FC input: Conv1 output pooled = 32 * 15 * 15 = 7200
FC_IN = C1_OUT * poolH1 * poolW1

np.random.seed(42)
std1 = np.sqrt(2.0 / (C1_IN * KH1 * KW1))
w1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * std1 * 0.1
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / FC_IN) * 0.1
fc_b = np.zeros(10, dtype=np.float32)

dw1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
dfw = lib.gpu_malloc(10 * FC_IN * 4)
dfb = lib.gpu_malloc(40)
lib.gpu_memcpy_h2d(dw1, w1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(dfw, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(dfb, fc_b.ctypes.data, 40)

data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:BATCH].astype(np.float32) / 255.0
    y = np.array(batch[b"labels"][:BATCH])
    x = x.reshape(-1, 3, 32, 32)

dx = lib.gpu_malloc(BATCH * C1_IN * H * W * 4)
lib.gpu_memcpy_h2d(dx, x.ctypes.data, BATCH * C1_IN * H * W * 4)

dc1_col = lib.gpu_malloc(C1_IN * KH1 * KW1 * BATCH * outH1 * outW1 * 4)
dc1r = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
lib.im2col_forward(dx, dc1_col, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
lib.gemm_forward(dw1, dc1_col, dc1r, C1_OUT, BATCH * outH1 * outW1, C1_IN * KH1 * KW1)
lib.leaky_relu_forward(dc1r, c_float(0.1), C1_OUT * BATCH * outH1 * outW1)

dc1 = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
lib.reorganize_forward(dc1r, dc1, BATCH, C1_OUT, outH1, outW1)
dp1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
dmi1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
lib.maxpool_forward_store(dp1, dc1, dmi1, BATCH, C1_OUT, outH1, outW1)

# Check pool1 values
hp1 = np.zeros(C1_OUT * BATCH * poolH1 * poolW1, dtype=np.float32)
lib.gpu_memcpy_d2h(hp1.ctypes.data, dp1, C1_OUT * BATCH * poolH1 * poolW1 * 4)
print(f"Pool1: min={hp1.min():.4f} max={hp1.max():.4f}")

# FC
dfi = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(dfi, hp1.ctypes.data, BATCH * FC_IN * 4)
dfo = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(dfi, dfw, dfb, dfo, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, dfo, BATCH * 10 * 4)
print(f"FC out: min={h_out.min():.4f} max={h_out.max():.4f}")

h_sh = h_out - h_out.max(axis=1, keepdims=True)
exp_o = np.exp(h_sh)
probs = exp_o / exp_o.sum(axis=1, keepdims=True)
loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
print(f"Loss: {loss:.4f}")

# Backward
labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot

grad_fc_w = d_loss.T @ hp1.reshape(BATCH, FC_IN)
grad_fc_w_c = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)

dgfw = lib.gpu_malloc(10 * FC_IN * 4)
lib.gpu_memcpy_h2d(dgfw, grad_fc_w_c.ctypes.data, 10 * FC_IN * 4)
lib.apply_sgd_update(dfw, dgfw, c_float(LR), 10 * FC_IN)

h_fc_w = np.zeros(10 * FC_IN, dtype=np.float32)
lib.gpu_memcpy_d2h(h_fc_w.ctypes.data, dfw, 10 * FC_IN * 4)
print(f"fc_w after: min={h_fc_w.min():.4f} max={h_fc_w.max():.4f}")

lib.gpu_free(dx); lib.gpu_free(dc1_col); lib.gpu_free(dc1r); lib.gpu_free(dc1)
lib.gpu_free(dp1); lib.gpu_free(dmi1); lib.gpu_free(dfi); lib.gpu_free(dfo)
lib.gpu_free(dgfw); lib.gpu_free(dw1); lib.gpu_free(dfw); lib.gpu_free(dfb)
print("DONE")
