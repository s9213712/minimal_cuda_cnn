#!/usr/bin/env python3
"""Check GEMM output layout"""
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
    x = batch[b"data"][:8].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W, KH, KW, OC = 8, 3, 32, 32, 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 2.0

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv, OC, BATCH * outH * outW, C * KH * KW)
lib.apply_relu(d_conv, OC * BATCH * outH * outW)

all_conv = np.zeros((OC * BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(all_conv.ctypes.data, d_conv, OC * BATCH * outH * outW * 4)

# If layout is (BATCH*outH*outW, OC), then each group of OC elements is one spatial point
# Check first spatial point (indices 0-31)
print("If layout is (N_spatial, OC):")
print("First spatial point (all channels):", all_conv[0:32].tolist())
print("Second spatial point:", all_conv[32:64].tolist())

# Check first few spatial points for non-zero in any channel
for sp in range(10):
    vals = all_conv[sp*32:(sp+1)*32]
    max_val = vals.max()
    if max_val > 0:
        print(f"  Spatial point {sp}: max={max_val:.4f}")

# If layout is (OC, BATCH*outH*outW), check first channel
print("\nIf layout is (OC, N_spatial):")
ch0 = all_conv[0::32]  # Every 32nd element starting from 0
print("Channel 0 at spatial points 0-9:", ch0[:10].tolist())
ch1 = all_conv[1::32]  # Every 32nd element starting from 1
print("Channel 1 at spatial points 0-9:", ch1[:10].tolist())

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv)
