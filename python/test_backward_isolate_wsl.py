#!/usr/bin/env python3
"""Isolate which backward kernel causes NaN - WSL version"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os
import sys

# Adjust path for WSL
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
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

BATCH = 4

# Architecture
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

print(f"=== Architecture ===")
print(f"Conv1: {C1_IN}→{C1_OUT}, out={outH1}x{outW1}, pool={poolH1}x{poolW1}")
print(f"Conv2: {C2_IN}→{C2_OUT}, out={outH2}x{outW2}, pool={poolH2}x{poolW2}")
print(f"FC_IN: {FC_IN}")

np.random.seed(42)
std1 = 0.01
std2 = 0.01
w_conv1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * std1
w_conv2 = np.random.randn(C2_OUT * C2_IN * KH2 * KW2).astype(np.float32) * std2
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * 0.01
fc_b = np.zeros(10, dtype=np.float32)

x = np.random.randn(BATCH, C1_IN, H, W).astype(np.float32) * 0.5
y = np.array([1, 2, 3, 5])[:BATCH]

# GPU weights - allocate ONCE like training
d_w_conv1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
d_w_conv2 = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)

fc_w_orig = fc_w.copy()

def check(name, arr, label="min/max"):
    nan = np.isnan(arr).sum()
    inf = np.isinf(arr).sum()
    print(f"  {name}: {label}={arr.min():.4f}/{arr.max():.4f}, nan={nan}, inf={inf}")
    return nan == 0 and inf == 0

def g2h(arr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, arr, size * 4)
    return h

# ========== FORWARD ==========
print("\n=== FORWARD ===")
d_x = lib.gpu_malloc(BATCH * C1_IN * H * W * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C1_IN * H * W * 4)

# Conv1
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

# Conv2
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

# FC
h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, BATCH * FC_IN * 4)
check("pool2", h_pool2)

d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH * FC_IN * 4)

d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
check("fc_out", h_out)

h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
exp_out = np.exp(h_out_shifted)
probs = exp_out / exp_out.sum(axis=1, keepdims=True)
loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
print(f"  Loss: {loss:.4f}")
check("probs", probs)

# ========== BACKWARD ==========
print("\n=== BACKWARD ===")

labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot
check("d_loss", d_loss)

grad_fc_w = d_loss.T @ h_pool2
check("grad_fc_w", grad_fc_w)

grad_pool2 = d_loss @ fc_w.reshape(10, FC_IN)
check("grad_pool2", grad_pool2)

# SGD update fc_w
grad_fc_w_clipped = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
d_fc_grad_w = lib.gpu_malloc(10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_grad_w, grad_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(0.001), 10 * FC_IN)

# Clip fc_w
h_fc_w = g2h(d_fc_w, 10 * FC_IN)
h_fc_w_clipped = np.clip(h_fc_w, -1.0, 1.0).astype(np.float32)
lib.gpu_memcpy_h2d(d_fc_w, h_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
check("fc_w after SGD", h_fc_w_clipped)

# ========== POOL2 BACKWARD ==========
print("\n--- pool2 backward ---")
d_pool2_grad = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2.flatten().ctypes.data, C2_OUT * BATCH * poolH2 * poolW2 * 4)

d_conv2_grad = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
lib.gpu_memset(d_conv2_grad, 0, C2_OUT * BATCH * outH2 * outW2 * 4)

# Check max_idx validity
max_idx2 = np.zeros(C2_OUT * BATCH * poolH2 * poolW2, dtype=np.int32)
lib.gpu_memcpy_d2h(max_idx2.ctypes.data, d_max_idx2, max_idx2.size * 4)
valid_range = C2_OUT * BATCH * outH2 * outW2
bad_idx = np.sum((max_idx2 < 0) | (max_idx2 >= valid_range))
print(f"  max_idx2 range: {max_idx2.min()}-{max_idx2.max()}, valid=0-{valid_range-1}, bad={bad_idx}")

lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)

h_conv2_grad = g2h(d_conv2_grad, C2_OUT * BATCH * outH2 * outW2)
ok = check("conv2_grad after pool2_bwd", h_conv2_grad)
if not ok:
    print("  *** pool2_backward_use_idx CORRUPTED ***")

# ========== WEIGHT CORRUPTION CHECK ==========
print("\n=== WEIGHT CORRUPTION CHECK ===")
h_w1 = g2h(d_w_conv1, C1_OUT * C1_IN * KH1 * KW1)
h_w2 = g2h(d_w_conv2, C2_OUT * C2_IN * KH2 * KW2)
h_fc = g2h(d_fc_w, 10 * FC_IN)

ok1 = check("w_conv1 (GPU)", h_w1)
ok2 = check("w_conv2 (GPU)", h_w2)
ok3 = check("fc_w (GPU)", h_fc)

print(f"\n{'ALL OK' if ok1 and ok2 and ok3 else 'CORRUPTION DETECTED'}")
