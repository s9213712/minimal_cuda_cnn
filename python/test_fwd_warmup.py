#!/usr/bin/env python3
"""Minimal test - forward only, same as test_step.py"""
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
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

BATCH = 64
C1_IN = 3; C1_OUT = 32; H = 32; W = 32; KH1 = 3; KW1 = 3
outH1 = 30; outW1 = 30; poolH1 = 15; poolW1 = 15
C2_IN = 32; C2_OUT = 64; KH2 = 3; KW2 = 3
outH2 = 13; outW2 = 13; poolH2 = 6; poolW2 = 6
FC_IN = 2304

# Warmup FIRST (like train scripts do)
dummy = lib.gpu_malloc(4)
lib.gpu_free(dummy)

np.random.seed(42)
std1 = np.sqrt(2.0 / (C1_IN * KH1 * KW1))
std2 = np.sqrt(2.0 / (C2_IN * KH2 * KW2))
w1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * std1 * 0.1
w2 = np.random.randn(C2_OUT * C2_IN * KH2 * KW2).astype(np.float32) * std2 * 0.1
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / FC_IN) * 0.1
fc_b = np.zeros(10, dtype=np.float32)

dw1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
dw2 = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
dfw = lib.gpu_malloc(10 * FC_IN * 4)
dfb = lib.gpu_malloc(40)
lib.gpu_memcpy_h2d(dw1, w1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(dw2, w2.ctypes.data, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memcpy_h2d(dfw, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(dfb, fc_b.ctypes.data, 40)

data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:BATCH].astype(np.float32) / 255.0
    y = np.array(batch[b"labels"][:BATCH])
    x = x.reshape(-1, 3, 32, 32)

# FORWARD ONLY (like test_step.py but with warmup first)
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

dc2_col = lib.gpu_malloc(C2_IN * KH2 * KW2 * BATCH * outH2 * outW2 * 4)
dc2r = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
lib.im2col_forward(dp1, dc2_col, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
lib.gemm_forward(dw2, dc2_col, dc2r, C2_OUT, BATCH * outH2 * outW2, C2_IN * KH2 * KW2)
lib.leaky_relu_forward(dc2r, c_float(0.1), C2_OUT * BATCH * outH2 * outW2)

dc2 = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
lib.reorganize_forward(dc2r, dc2, BATCH, C2_OUT, outH2, outW2)

dp2 = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
dmi2 = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
lib.maxpool_forward_store(dp2, dc2, dmi2, BATCH, C2_OUT, outH2, outW2)

h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool2.ctypes.data, dp2, BATCH * FC_IN * 4)
print(f"h_pool2: min={h_pool2.min():.4f} max={h_pool2.max():.4f}")

dfi = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(dfi, h_pool2.ctypes.data, BATCH * FC_IN * 4)
dfo = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(dfi, dfw, dfb, dfo, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, dfo, BATCH * 10 * 4)
print(f"FC out: min={h_out.min():.4f} max={h_out.max():.4f}")

print("=== ALL OK ===")
