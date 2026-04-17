#!/usr/bin/env python3
"""Check pool grad at specific locations"""
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
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_nchw.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:64].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W = 64, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1
FC_IN = OC * 15 * 15

np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.3

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
lib.leaky_relu_forward(d_conv_raw, c_float(0.02), OC * BATCH * outH * outW)

d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

# Check pool outputs
h_pool = np.zeros((BATCH, OC, 15, 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print("Pool output stats:")
print(f"  min={h_pool.min():.6f}, max={h_pool.max():.6f}")
print(f"  mean={h_pool.mean():.6f}, std={h_pool.std():.6f}")
print(f"  zeros: {(h_pool == 0).sum()} / {h_pool.size}")

# For a specific batch and channel, check how many non-zeros
for b in range(3):
    non_zero = (h_pool[b] != 0).sum()
    print(f"  batch {b}: {non_zero}/{OC*15*15} non-zero pool outputs")

# Create arbitrary gradient for pool grad
grad_pool = np.random.randn(BATCH, FC_IN).astype(np.float32) * 0.1
print("\nPool grad stats:")
print(f"  min={grad_pool.min():.6f}, max={grad_pool.max():.6f}")

d_pool_grad = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.gpu_memcpy_h2d(d_pool_grad, grad_pool.flatten().ctypes.data, OC * BATCH * 15 * 15 * 4)

d_conv_grad = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gpu_memset(d_conv_grad, 0, OC * BATCH * outH * outW * 4)
lib.maxpool_backward_nchw(d_pool_grad, d_conv, d_conv_grad, BATCH, OC, outH, outW)

h_conv_grad = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_grad.ctypes.data, d_conv_grad, BATCH * OC * outH * outW * 4)
print("\nConv grad stats after pool backward:")
print(f"  min={h_conv_grad.min():.6f}, max={h_conv_grad.max():.6f}")
print(f"  mean={h_conv_grad.mean():.6f}, std={h_conv_grad.std():.6f}")
print(f"  zeros: {(h_conv_grad == 0).sum()} / {h_conv_grad.size}")

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
lib.gpu_free(d_pool_grad)
lib.gpu_free(d_conv_grad)
print("\nDone!")
