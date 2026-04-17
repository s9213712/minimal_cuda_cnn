#!/usr/bin/env python3
"""Debug pool output values"""
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
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:64].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W = 64, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)

h_conv_raw = np.zeros((OC, BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_raw.ctypes.data, d_conv_raw, OC * BATCH * outH * outW * 4)
print("GEMM output (before leaky ReLU):")
print(f"  min={h_conv_raw.min():.6f}, max={h_conv_raw.max():.6f}")
print(f"  mean={h_conv_raw.mean():.6f}, std={h_conv_raw.std():.6f}")
print(f"  % negative: {(h_conv_raw < 0).sum() / h_conv_raw.size * 100:.1f}%")

# Apply leaky ReLU
lib.leaky_relu_forward(d_conv_raw, c_float(0.01), OC * BATCH * outH * outW)

h_conv_after = np.zeros((OC, BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_after.ctypes.data, d_conv_raw, OC * BATCH * outH * outW * 4)
print("\nAfter Leaky ReLU:")
print(f"  min={h_conv_after.min():.6f}, max={h_conv_after.max():.6f}")
print(f"  mean={h_conv_after.mean():.6f}, std={h_conv_after.std():.6f}")
print(f"  % very small (<0.001): {(np.abs(h_conv_after) < 0.001).sum() / h_conv_after.size * 100:.1f}%")

# Reorganize
d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

# Pool
d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

h_pool = np.zeros((BATCH, OC, 15, 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print("\nPool output:")
print(f"  min={h_pool.min():.6f}, max={h_pool.max():.6f}")
print(f"  mean={h_pool.mean():.6f}, std={h_pool.std():.6f}")
print(f"  % zeros: {(h_pool == 0).sum() / h_pool.size * 100:.1f}%")

# Check specific values
print(f"\n  Sample values (batch 0, ch 0):")
print(h_pool[0, 0, :, :])

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
