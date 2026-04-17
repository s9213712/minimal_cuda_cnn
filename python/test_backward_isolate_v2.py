#!/usr/bin/env python3
"""Isolate which backward kernel causes NaN"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os

so = os.path.join(os.path.dirname(__file__), "..", "cpp", "libminimal_cuda_cnn.so")
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

BATCH = 4  # small for speed

# Architecture
C1_IN, C1_OUT = 3, 32
H, W = 32, 32
KH1, KW1 = 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1  # 30
poolH1, poolW1 = outH1 // 2, outW1 // 2   # 15

C2_IN, C2_OUT = 32, 64
KH2, KW2 = 3, 3
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1  # 13
poolH2, poolW2 = outH2 // 2, outW2 // 2   # 6

FC_IN = C2_OUT * poolH2 * poolW2  # 64*6*6 = 2304

print(f"=== Architecture ===")
print(f"Conv1: {C1_IN}→{C1_OUT}, {KH1}×{KW1}, out={outH1}×{outW1}, pool={poolH1}×{poolW1}")
print(f"Conv2: {C2_IN}→{C2_OUT}, {KH2}×{KW2}, out={outH2}×{outW2}, pool={poolH2}×{poolW2}")
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

# GPU weights - allocate ONCE like training does
d_w_conv1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
d_w_conv2 = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)

# Save original weights for comparison
w_conv1_orig = w_conv1.copy()
w_conv2_orig = w_conv2.copy()
fc_w_orig = fc_w.copy()

def check(name, arr, label="min/max"):
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    print(f"  {name}: {label}={arr.min():.4f}/{arr.max():.4f}, nan={nan_count}, inf={inf_count}")
    return nan_count == 0 and inf_count == 0

def gpu_to_h(arr):
    h = np.zeros(arr.size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, arr, arr.size * 4)
    return h

# ============== FORWARD ==============
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
check("pool2 (h)", h_pool2)

d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH * FC_IN * 4)

d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
check("fc_out (h)", h_out)

# Softmax
h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
exp_out = np.exp(h_out_shifted)
probs = exp_out / exp_out.sum(axis=1, keepdims=True)
loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
print(f"  Loss: {loss:.4f}")
check("probs", probs)

# ============== BACKWARD ==============
print("\n=== BACKWARD ===")

# FC gradient
labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot
check("d_loss", d_loss)

grad_fc_w = d_loss.T @ h_pool2  # (10, 2304)
check("grad_fc_w", grad_fc_w, "min/max/mean")

# Pool2 gradient (using ORIGINAL fc_w)
fc_w_reshaped = fc_w.reshape(10, FC_IN)
grad_pool2 = d_loss @ fc_w_reshaped  # (4, 2304)
check("grad_pool2", grad_pool2, "min/max/mean")

# d_fc_w SGD update
grad_fc_w_clipped = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
d_fc_grad_w = lib.gpu_malloc(10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_grad_w, grad_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(0.001), 10 * FC_IN)

# Clip fc_w
h_fc_w = np.zeros(10 * FC_IN, dtype=np.float32)
lib.gpu_memcpy_d2h(h_fc_w.ctypes.data, d_fc_w, 10 * FC_IN * 4)
h_fc_w_clipped = np.clip(h_fc_w, -1.0, 1.0).astype(np.float32)
lib.gpu_memcpy_h2d(d_fc_w, h_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
check("fc_w after SGD", h_fc_w_clipped)

# Check pool2 backward
print("\n--- pool2 backward ---")
d_pool2_grad = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2.flatten().ctypes.data, C2_OUT * BATCH * poolH2 * poolW2 * 4)

d_conv2_grad = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
lib.gpu_memset(d_conv2_grad, 0, C2_OUT * BATCH * outH2 * outW2 * 4)

h_max_idx2_before = np.zeros(C2_OUT * BATCH * poolH2 * poolW2, dtype=np.int32)
lib.gpu_memcpy_d2h(h_max_idx2_before.ctypes.data, d_max_idx2, C2_OUT * BATCH * poolH2 * poolW2 * 4)
print(f"  max_idx2 range: {h_max_idx2_before.min()}/{h_max_idx2_before.max()} (valid: 0-{C2_OUT*BATCH*outH2*outW2-1})")

lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)

h_conv2_grad = np.zeros(C2_OUT * BATCH * outH2 * outW2, dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv2_grad.ctypes.data, d_conv2_grad, C2_OUT * BATCH * outH2 * outW2 * 4)
ok = check("conv2_grad after pool2_bwd", h_conv2_grad)
if not ok:
    print("  *** pool2_backward_use_idx CORRUPTED conv2_grad ***")

# ============== CHECK WEIGHTS ==============
print("\n=== WEIGHT CORRUPTION CHECK ===")
h_w_conv1_after = np.zeros(C1_OUT * C1_IN * KH1 * KW1, dtype=np.float32)
lib.gpu_memcpy_d2h(h_w_conv1_after.ctypes.data, d_w_conv1, C1_OUT * C1_IN * KH1 * KW1 * 4)
ok1 = check("w_conv1 (GPU)", h_w_conv1_after)

h_w_conv2_after = np.zeros(C2_OUT * C2_IN * KH2 * KW2, dtype=np.float32)
lib.gpu_memcpy_d2h(h_w_conv2_after.ctypes.data, d_w_conv2, C2_OUT * C2_IN * KH2 * KW2 * 4)
ok2 = check("w_conv2 (GPU)", h_w_conv2_after)

h_fc_w_after = np.zeros(10 * FC_IN, dtype=np.float32)
lib.gpu_memcpy_d2h(h_fc_w_after.ctypes.data, d_fc_w, 10 * FC_IN * 4)
ok3 = check("fc_w (GPU)", h_fc_w_after)

print(f"\n{'✅ All OK' if ok1 and ok2 and ok3 else '❌ CORRUPTION DETECTED'}")
