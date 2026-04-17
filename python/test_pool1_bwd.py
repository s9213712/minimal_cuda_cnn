#!/usr/bin/env python3
"""Test pool1 backward"""
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
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH = 64
LR = 0.001
C1_IN = 3; C1_OUT = 32; H = 32; W = 32; KH1 = 3; KW1 = 3
outH1 = 30; outW1 = 30; poolH1 = 15; poolW1 = 15

np.random.seed(42)
w1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * 0.05
dw1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(dw1, w1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)

data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:BATCH].astype(np.float32) / 255.0
    x = x.reshape(-1, 3, 32, 32)

# FORWARD
dx = lib.gpu_malloc(BATCH * C1_IN * H * W * 4)
dc1_col = lib.gpu_malloc(C1_IN * KH1 * KW1 * BATCH * outH1 * outW1 * 4)
dc1r = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
dc1 = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
dp1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
dmi1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)

lib.gpu_memcpy_h2d(dx, x.ctypes.data, BATCH * C1_IN * H * W * 4)
lib.im2col_forward(dx, dc1_col, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
lib.gemm_forward(dw1, dc1_col, dc1r, C1_OUT, BATCH * outH1 * outW1, C1_IN * KH1 * KW1)
lib.leaky_relu_forward(dc1r, c_float(0.1), C1_OUT * BATCH * outH1 * outW1)
lib.reorganize_forward(dc1r, dc1, BATCH, C1_OUT, outH1, outW1)
lib.maxpool_forward_store(dp1, dc1, dmi1, BATCH, C1_OUT, outH1, outW1)

# Check pool1 values
hp1 = np.zeros(C1_OUT * BATCH * poolH1 * poolW1, dtype=np.float32)
lib.gpu_memcpy_d2h(hp1.ctypes.data, dp1, C1_OUT * BATCH * poolH1 * poolW1 * 4)
print(f"Pool1 output: min={hp1.min():.4f} max={hp1.max():.4f}")

# Create fake gradient for pool1
grad_pool1 = np.random.randn(C1_OUT * BATCH * poolH1 * poolW1).astype(np.float32) * 0.1
print(f"grad_pool1: min={grad_pool1.min():.4f} max={grad_pool1.max():.4f}")

d_grad_pool1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
lib.gpu_memcpy_h2d(d_grad_pool1, grad_pool1.ctypes.data, C1_OUT * BATCH * poolH1 * poolW1 * 4)

# Alloc gradient for conv1 output
d_conv1_grad = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
lib.gpu_memset(d_conv1_grad, 0, C1_OUT * BATCH * outH1 * outW1 * 4)

print(f"Calling maxpool_backward_use_idx with:")
print(f"  d_grad_pool1={d_grad_pool1}")
print(f"  dmi1={dmi1}")
print(f"  d_conv1_grad={d_conv1_grad}")
print(f"  BATCH={BATCH}, C1_OUT={C1_OUT}, outH1={outH1}, outW1={outW1}")

lib.maxpool_backward_use_idx(d_grad_pool1, dmi1, d_conv1_grad, BATCH, C1_OUT, outH1, outW1)

h_conv1_grad = np.zeros(C1_OUT * BATCH * outH1 * outW1, dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv1_grad.ctypes.data, d_conv1_grad, C1_OUT * BATCH * outH1 * outW1 * 4)
print(f"d_conv1_grad: min={h_conv1_grad.min():.4f} max={h_conv1_grad.max():.4f}")
print(f"Non-zero count: {np.count_nonzero(h_conv1_grad)}")

lib.gpu_free(dx); lib.gpu_free(dc1_col); lib.gpu_free(dc1r); lib.gpu_free(dc1)
lib.gpu_free(dp1); lib.gpu_free(dmi1); lib.gpu_free(d_grad_pool1); lib.gpu_free(d_conv1_grad)
lib.gpu_free(dw1)
print("=== DONE ===")
