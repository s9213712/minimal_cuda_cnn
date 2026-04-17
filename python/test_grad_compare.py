#!/usr/bin/env python3
"""Debug: Compare conv_backward's input grad vs gemm+im2col chain"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os, pickle

workspace = "/mnt/c/Users/user/.openclaw/workspace"
so = os.path.join(workspace, "NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)

lib.gpu_malloc.argtypes = [ctypes.c_size_t]; lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.im2col_backward.argtypes = [c_void_p]*10
lib.gemm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

BATCH = 8  # small for debug
C2_IN, C2_OUT = 32, 64; KH2, KW2 = 3, 3
poolH1, poolW1 = 15, 15
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2

# Init
np.random.seed(42)
w_conv2 = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * 0.05
pool1 = np.random.randn(BATCH*C2_IN*poolH1*poolW1).astype(np.float32) * 0.1
d_c2g = np.random.randn(C2_OUT*BATCH*outH2*outW2).astype(np.float32) * 0.01

d_wc2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_p1 = lib.gpu_malloc(BATCH*C2_IN*poolH1*poolW1*4)
d_c2g_d = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
lib.gpu_memcpy_h2d(d_wc2, w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memcpy_h2d(d_p1, pool1.ctypes.data, BATCH*C2_IN*poolH1*poolW1*4)
lib.gpu_memcpy_h2d(d_c2g_d, d_c2g.ctypes.data, C2_OUT*BATCH*outH2*outW2*4)

# Method 1: conv_backward gives us d_pool1_grad
d_p1_grad_m1 = lib.gpu_malloc(BATCH*C2_IN*poolH1*poolW1*4)
d_wc2_grad_m1 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memset(d_p1_grad_m1, 0, BATCH*C2_IN*poolH1*poolW1*4)
lib.gpu_memset(d_wc2_grad_m1, 0, C2_OUT*C2_IN*KH2*KW2*4)
lib.conv_backward(d_c2g_d, d_p1, d_wc2, d_wc2_grad_m1,
                  BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
h_p1_grad_m1 = g2h(d_p1_grad_m1, BATCH*C2_IN*poolH1*poolW1)
print(f"Method1 (conv_backward input grad): min={h_p1_grad_m1.min():.6f}, max={h_p1_grad_m1.max():.6f}, mean={h_p1_grad_m1.mean():.6f}")

# Method 2: gemm_backward + im2col_backward
# im2col forward: d_col2 = im2col(d_p1)
d_col2 = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
lib.im2col_forward(d_p1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)

# gemm_backward: d_col2_grad = d_c2g @ w_conv2^T
d_col2_grad = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
d_c2g_copy = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
lib.gpu_memcpy_h2d(d_c2g_copy, d_c2g.ctypes.data, C2_OUT*BATCH*outH2*outW2*4)
lib.gpu_memset(d_col2_grad, 0, C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
lib.gemm_backward(d_c2g_d, d_wc2, d_col2, d_c2g_copy, d_col2_grad,
                  C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
h_col2_grad = g2h(d_col2_grad, C2_IN*KH2*KW2*BATCH*outH2*outW2)
print(f"Method2a (gemm_backward col2_grad): min={h_col2_grad.min():.6f}, max={h_col2_grad.max():.6f}, mean={h_col2_grad.mean():.6f}")

# im2col_backward: d_pool1_grad = im2col_backward(d_col2_grad)
d_p1_grad_m2 = lib.gpu_malloc(BATCH*C2_IN*poolH1*poolW1*4)
lib.gpu_memset(d_p1_grad_m2, 0, BATCH*C2_IN*poolH1*poolW1*4)
lib.im2col_backward(d_col2_grad, d_p1_grad_m2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
h_p1_grad_m2 = g2h(d_p1_grad_m2, BATCH*C2_IN*poolH1*poolW1)
print(f"Method2b (im2col_backward pool1_grad): min={h_p1_grad_m2.min():.6f}, max={h_p1_grad_m2.max():.6f}, mean={h_p1_grad_m2.mean():.6f}")

# Compare
diff = np.abs(h_p1_grad_m1 - h_p1_grad_m2)
print(f"\nComparison:")
print(f"  M1: mean={h_p1_grad_m1.mean():.6f}, std={h_p1_grad_m1.std():.6f}")
print(f"  M2: mean={h_p1_grad_m2.mean():.6f}, std={h_p1_grad_m2.std():.6f}")
print(f"  Diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
print(f"  Correlation: {np.corrcoef(h_p1_grad_m1.flatten(), h_p1_grad_m2.flatten())[0,1]:.6f}")
print(f"  Same sign: {np.sum(np.sign(h_p1_grad_m1) == np.sign(h_p1_grad_m2))}/{h_p1_grad_m1.size}")

# FREE
lib.gpu_free(d_wc2); lib.gpu_free(d_p1); lib.gpu_free(d_c2g_d)
lib.gpu_free(d_p1_grad_m1); lib.gpu_free(d_wc2_grad_m1)
lib.gpu_free(d_col2); lib.gpu_free(d_col2_grad); lib.gpu_free(d_c2g_copy)
lib.gpu_free(d_p1_grad_m2)
print("\nDone!")
