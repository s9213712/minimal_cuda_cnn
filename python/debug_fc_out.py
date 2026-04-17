#!/usr/bin/env python3
"""Debug FC output"""
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
lib.apply_relu.argtypes = [c_void_p, c_int]
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:64].astype(np.float32) / 255.0
    y = np.array(batch[b"labels"][:64])
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W, KH, KW, OC = 64, 3, 32, 32, 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1
FC_IN = OC * 15 * 15

np.random.seed(42)
w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * np.sqrt(2.0 / (C * KH * KW + OC * KH * KW))
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / (FC_IN + 10))
fc_b = np.zeros(10, dtype=np.float32)

# Forward pass
d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w_conv = lib.gpu_malloc(OC * C * KH * KW * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
lib.gemm_forward(d_w_conv, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
lib.apply_relu(d_conv_raw, OC * BATCH * outH * outW)

d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

h_pool = np.zeros((BATCH, FC_IN), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * FC_IN * 4)
print("Pool output:", h_pool.shape)
print("  min:", h_pool.min(), "max:", h_pool.max())
print("  any NaN:", np.isnan(h_pool).any())

d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_in, h_pool.ctypes.data, BATCH * FC_IN * 4)

d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)

h_fc_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_fc_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
print("\nFC output:", h_fc_out.shape)
print("  min:", h_fc_out.min(), "max:", h_fc_out.max())
print("  any NaN:", np.isnan(h_fc_out).any())
print("  sample:", h_fc_out[0])

lib.gpu_free(d_x)
lib.gpu_free(d_w_conv)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
lib.gpu_free(d_fc_in)
lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
lib.gpu_free(d_fc_out)
