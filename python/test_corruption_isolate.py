#!/usr/bin/env python3
"""Isolate memory corruption: forward-only vs forward+backward, same random seed"""
import ctypes
import numpy as np
import sys
sys.path.insert(0, '/home/s92137/NN/minimal_cuda_cnn/python')
import libminimal_cuda_cnn as lib
from ctypes import c_float, c_int

BATCH = 64
C1_IN, C1_OUT = 3, 32
C2_IN, C2_OUT = 32, 64
H, W = 32, 32
KH1, KW1, KH2, KW2 = 3, 3, 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1
poolH1, poolW1 = outH1 // 2, outW1 // 2
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2
FC_IN = C2_OUT * poolH2 * poolW2

np.random.seed(42)
w_conv1 = np.random.randn(C1_OUT, C1_IN, KH1, KW1).astype(np.float32) * 0.01
w_conv2 = np.random.randn(C2_OUT, C2_IN, KH2, KW2).astype(np.float32) * 0.01
fc_w = np.random.randn(10, FC_IN).astype(np.float32) * 0.01
fc_b = np.zeros(10, dtype=np.float32)

d_w_conv1 = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_w_conv2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_fc_w = lib.gpu_malloc(10*FC_IN*4)
d_fc_b = lib.gpu_malloc(10*4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10*FC_IN*4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10*4)

def g2h(ptr, size):
    arr = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(arr.ctypes.data, ptr, size*4)
    return arr

import pickle
data_path = '/mnt/c/Users/user/.openclaw/workspace/NN/minimal_cuda_cnn/data/cifar-10-batches-py'
with open(f'{data_path}/data_batch_1', 'rb') as f:
    b = pickle.load(f, encoding='bytes')
x_all = b[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
y_all = np.array(b[b'labels'])

print("=== FORWARD ONLY TEST ===")
for batch_idx in range(5):
    idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
    x = x_all[idx_s:idx_e]; y = y_all[idx_s:idx_e]
    
    d_x = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
    lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH*C1_IN*H*W*4)
    
    d_col1 = lib.gpu_malloc(C1_IN*KH1*KW1*BATCH*outH1*outW1*4)
    d_conv1_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
    lib.im2col_forward(d_x, d_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
    lib.gemm_forward(d_w_conv1, d_col1, d_conv1_raw, C1_OUT, BATCH*outH1*outW1, C1_IN*KH1*KW1)
    lib.leaky_relu_forward(d_conv1_raw, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
    d_conv1 = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
    lib.reorganize_forward(d_conv1_raw, d_conv1, BATCH, C1_OUT, outH1, outW1)
    d_pool1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
    d_max_idx1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
    lib.maxpool_forward_store(d_pool1, d_conv1, d_max_idx1, BATCH, C1_OUT, outH1, outW1)
    
    d_col2 = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
    d_conv2_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
    lib.im2col_forward(d_pool1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
    lib.gemm_forward(d_w_conv2, d_col2, d_conv2_raw, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
    lib.leaky_relu_forward(d_conv2_raw, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
    d_conv2 = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
    lib.reorganize_forward(d_conv2_raw, d_conv2, BATCH, C2_OUT, outH2, outW2)
    d_pool2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
    d_max_idx2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
    lib.maxpool_forward_store(d_pool2, d_conv2, d_max_idx2, BATCH, C2_OUT, outH2, outW2)
    
    h_pool2_chk = np.zeros((BATCH, C2_OUT, poolH2, poolW2), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool2_chk.ctypes.data, d_pool2, BATCH*C2_OUT*poolH2*poolW2*4)
    print(f"[FWD_ONLY B{batch_idx}] pool2: max={np.abs(h_pool2_chk).max():.6e} nan={np.isnan(h_pool2_chk).sum()}", flush=True)
    
    # FC forward
    h_pool2 = h_pool2_chk.reshape(BATCH, -1)
    d_fc_in = lib.gpu_malloc(BATCH*FC_IN*4)
    lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH*FC_IN*4)
    d_fc_out = lib.gpu_malloc(BATCH*10*4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH*10*4)
    print(f"[FWD_ONLY B{batch_idx}] h_out: max={np.abs(h_out).max():.6e}", flush=True)
    
    # SGD update (forward-only style: compute grad from cross-entropy)
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
    d_loss = probs.copy()
    d_loss[np.arange(BATCH), y] -= 1
    d_loss /= BATCH
    grad_fc_w = (d_loss.T @ h_pool2) / BATCH
    
    h_fc_w_before = g2h(d_fc_w, 10*FC_IN)
    d_fc_grad_w = lib.gpu_malloc(10*FC_IN*4)
    lib.gpu_memcpy_h2d(d_fc_grad_w, np.clip(grad_fc_w, -5.0, 5.0).flatten().astype(np.float32).ctypes.data, 10*FC_IN*4)
    lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(0.005), 10*FC_IN)
    h_fc_w_after = g2h(d_fc_w, 10*FC_IN)
    print(f"[FWD_ONLY B{batch_idx}] fc_w: before={h_fc_w_before.max():.6e} after={h_fc_w_after.max():.6e}", flush=True)
    
    lib.gpu_free(d_fc_out); lib.gpu_free(d_fc_in); lib.gpu_free(d_pool2); lib.gpu_free(d_max_idx2)
    lib.gpu_free(d_conv2); lib.gpu_free(d_conv2_raw); lib.gpu_free(d_col2)
    lib.gpu_free(d_pool1); lib.gpu_free(d_max_idx1); lib.gpu_free(d_conv1); lib.gpu_free(d_conv1_raw)
    lib.gpu_free(d_col1); lib.gpu_free(d_x); lib.gpu_free(d_fc_grad_w)

print("\n=== ALLOC/FREE STRESS TEST (no computation) ===")
# Stress test: allocate and free many times, see if fc_w gets corrupted
for i in range(10):
    a = lib.gpu_malloc(10*FC_IN*4)
    b = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
    c = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
    lib.gpu_free(a); lib.gpu_free(b); lib.gpu_free(c)
    h_fc_w_chk = g2h(d_fc_w, 10*FC_IN)
    print(f"[STRESS {i}] fc_w: max={h_fc_w_chk.max():.6e}", flush=True)

print("\n=== DONE ===")
