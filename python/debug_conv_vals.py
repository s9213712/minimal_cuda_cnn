#!/usr/bin/env python3
"""Check conv output values"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int
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
lib.apply_relu.argtypes = [c_void_p, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:64].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W = 64, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

# With 0.05 std
np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05

print("Weight stats:", w.shape, "min", w.min(), "max", w.max())

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

h_col = np.zeros((C * KH * KW, BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_col.ctypes.data, d_col, col_size * 4)
print("im2col output:", h_col.shape, "min", h_col.min(), "max", h_col.max())

d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)

h_conv_raw = np.zeros((OC, BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_raw.ctypes.data, d_conv_raw, OC * BATCH * outH * outW * 4)
print("GEMM output (before ReLU):", h_conv_raw.shape, "min", h_conv_raw.min(), "max", h_conv_raw.max())

# Check first few values of first channel
print("First channel, first 10 values:", h_conv_raw[0, 0:10])

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
