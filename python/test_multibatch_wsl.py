#!/usr/bin/env python3
"""Simulate real training loop - multiple iterations with persistent GPU weights"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os

workspace = "/mnt/c/Users/user/.openclaw/workspace"
so = os.path.join(workspace, "NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so")
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
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH = 4
ITERS = 3  # small number to isolate first failure

C1_IN, C1_OUT = 3, 32
H, W = 32, 32
KH1, KW1 = 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1
poolH1, poolW1 = outH1 // 2, outW1 // 2

C2_IN, C2_OUT = 32, 64
KH2, KW2 = 3, 3
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2

FC_IN = C2_OUT * poolH2 * poolW2

np.random.seed(42)
std1, std2 = 0.01, 0.01
w_conv1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * std1
w_conv2 = np.random.randn(C2_OUT * C2_IN * KH2 * KW2).astype(np.float32) * std2
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * 0.01
fc_b = np.zeros(10, dtype=np.float32)

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

def check(name, arr):
    nan, inf = np.isnan(arr).sum(), np.isinf(arr).sum()
    ok = nan == 0 and inf == 0
    print(f"  {name}: {arr.min():.4f}/{arr.max():.4f}, nan={nan}, inf={inf} {'OK' if ok else 'BAD'}")
    return ok

# Allocate persistent GPU weights ONCE
d_w_conv1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
d_w_conv2 = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)

for iteration in range(ITERS):
    print(f"\n=== Iteration {iteration} ===")
    
    x = np.random.randn(BATCH, C1_IN, H, W).astype(np.float32) * 0.5
    y = np.array([1, 2, 3, 5])[:BATCH]
    
    # ========== FORWARD ==========
    d_x = lib.gpu_malloc(BATCH * C1_IN * H * W * 4)
    lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C1_IN * H * W * 4)
    
    d_col1 = lib.gpu_malloc(C1_IN * KH1 * KW1 * BATCH * outH1 * outW1 * 4)
    d_conv1_raw = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
    lib.im2col_forward(d_x, d_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
    lib.gemm_forward(d_w_conv1, d_col1, d_conv1_raw, C1_OUT, BATCH * outH1 * outW1, C1_IN * KH1 * KW1)
    lib.leaky_relu_forward(d_conv1_raw, c_float(0.1), C1_OUT * BATCH * outH1 * outW1)
    d_conv1 = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
    lib.reorganize_forward(d_conv1_raw, d_conv1, BATCH, C1_OUT, outH1, outW1)
    d_pool1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
    d_max_idx1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
    lib.maxpool_forward_store(d_pool1, d_conv1, d_max_idx1, BATCH, C1_OUT, outH1, outW1)
    
    d_col2 = lib.gpu_malloc(C2_IN * KH2 * KW2 * BATCH * outH2 * outW2 * 4)
    d_conv2_raw = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
    lib.im2col_forward(d_pool1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
    lib.gemm_forward(d_w_conv2, d_col2, d_conv2_raw, C2_OUT, BATCH * outH2 * outW2, C2_IN * KH2 * KW2)
    lib.leaky_relu_forward(d_conv2_raw, c_float(0.1), C2_OUT * BATCH * outH2 * outW2)
    d_conv2 = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
    lib.reorganize_forward(d_conv2_raw, d_conv2, BATCH, C2_OUT, outH2, outW2)
    d_pool2 = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
    d_max_idx2 = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
    lib.maxpool_forward_store(d_pool2, d_conv2, d_max_idx2, BATCH, C2_OUT, outH2, outW2)
    
    h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, BATCH * FC_IN * 4)
    if not check("pool2", h_pool2):
        print("  *** BREAKING: pool2 corrupted ***")
        break
    
    d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
    lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH * FC_IN * 4)
    d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
    if not check("fc_out", h_out):
        print("  *** BREAKING: fc_out corrupted ***")
        break
    
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    exp_out = np.exp(h_out_shifted)
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    print(f"  Loss: {loss:.4f}")
    
    # ========== BACKWARD ==========
    labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
    labels_onehot[np.arange(BATCH), y] = 1.0
    d_loss = probs - labels_onehot
    
    grad_fc_w = d_loss.T @ h_pool2
    grad_pool2 = d_loss @ fc_w.reshape(10, FC_IN)
    grad_pool2_clipped = np.clip(grad_pool2, -1.0, 1.0)
    
    # SGD update fc_w
    d_fc_grad_w = lib.gpu_malloc(10 * FC_IN * 4)
    lib.gpu_memcpy_h2d(d_fc_grad_w, grad_fc_w.flatten().astype(np.float32).ctypes.data, 10 * FC_IN * 4)
    lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(0.001), 10 * FC_IN)
    h_fc_w = g2h(d_fc_w, 10 * FC_IN)
    h_fc_w_clipped = np.clip(h_fc_w, -1.0, 1.0).astype(np.float32)
    lib.gpu_memcpy_h2d(d_fc_w, h_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
    fc_w = h_fc_w_clipped.copy()
    
    if not check("fc_w after SGD", fc_w):
        print("  *** BREAKING: fc_w corrupted ***")
        break
    
    # Pool2 backward
    d_pool2_grad = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
    lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2_clipped.flatten().astype(np.float32).ctypes.data, C2_OUT * BATCH * poolH2 * poolW2 * 4)
    d_conv2_grad = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
    lib.gpu_memset(d_conv2_grad, 0, C2_OUT * BATCH * outH2 * outW2 * 4)
    lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)
    h_conv2_grad = g2h(d_conv2_grad, C2_OUT * BATCH * outH2 * outW2)
    if not check("conv2_grad", h_conv2_grad):
        print("  *** BREAKING: conv2_grad corrupted (POOL2 BACKWARD FAILS) ***")
        break
    
    # Check if weights survived
    h_w1 = g2h(d_w_conv1, C1_OUT * C1_IN * KH1 * KW1)
    h_w2 = g2h(d_w_conv2, C2_OUT * C2_IN * KH2 * KW2)
    h_fc = g2h(d_fc_w, 10 * FC_IN)
    if not check("w_conv1", h_w1) or not check("w_conv2", h_w2) or not check("fc_w", h_fc):
        print("  *** BREAKING: weights corrupted ***")
        break
    
    # FREE all
    lib.gpu_free(d_x); lib.gpu_free(d_col1); lib.gpu_free(d_conv1_raw); lib.gpu_free(d_conv1)
    lib.gpu_free(d_pool1); lib.gpu_free(d_max_idx1); lib.gpu_free(d_col2); lib.gpu_free(d_conv2_raw)
    lib.gpu_free(d_conv2); lib.gpu_free(d_pool2); lib.gpu_free(d_max_idx2); lib.gpu_free(d_fc_in)
    lib.gpu_free(d_fc_out); lib.gpu_free(d_fc_grad_w); lib.gpu_free(d_pool2_grad); lib.gpu_free(d_conv2_grad)
    print(f"  Iter {iteration} complete")

print("\n=== DONE ===")
h_w1 = g2h(d_w_conv1, C1_OUT * C1_IN * KH1 * KW1)
h_w2 = g2h(d_w_conv2, C2_OUT * C2_IN * KH2 * KW2)
h_fc = g2h(d_fc_w, 10 * FC_IN)
check("Final w_conv1", h_w1)
check("Final w_conv2", h_w2)
check("Final fc_w", h_fc)

lib.gpu_free(d_w_conv1); lib.gpu_free(d_w_conv2); lib.gpu_free(d_fc_w); lib.gpu_free(d_fc_b)
