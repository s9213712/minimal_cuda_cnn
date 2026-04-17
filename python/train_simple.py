#!/usr/bin/env python3
"""Simple working train script"""
import ctypes
import numpy as np
import time
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
lib.leaky_relu_backward.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH = 32
LR = 0.01
LEAKY_ALPHA = 0.1
C, H, W = 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1
FC_IN = OC * 15 * 15

print(f"Config: BATCH={BATCH}, LR={LR}")

# Initialize weights
np.random.seed(42)
w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * 0.05
fc_b = np.zeros(10, dtype=np.float32)

# GPU memory
d_w_conv = lib.gpu_malloc(OC * C * KH * KW * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)

# Load data
with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x_all = batch[b"data"][:10000].astype(np.float32) / 255.0
    y_all = np.array(batch[b"labels"][:10000])
x_all = x_all.reshape(-1, 3, 32, 32)
print(f"Data: {x_all.shape}, labels: {y_all.shape}")

# Training
NBATCHES = 50
print(f"Training {NBATCHES} batches...")
t0 = time.time()
for batch_idx in range(NBATCHES):
    idx_start = batch_idx * BATCH
    idx_end = idx_start + BATCH
    x = x_all[idx_start:idx_end]
    y = y_all[idx_start:idx_end]
    
    # Forward pass
    d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
    lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    
    d_col = lib.gpu_malloc(C * KH * KW * BATCH * outH * outW * 4)
    d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
    lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    lib.gemm_forward(d_w_conv, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
    lib.leaky_relu_forward(d_conv_raw, c_float(LEAKY_ALPHA), OC * BATCH * outH * outW)
    
    d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
    lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)
    
    d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
    d_max_idx = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
    lib.maxpool_forward_store(d_pool, d_conv, d_max_idx, BATCH, OC, outH, outW)
    
    h_pool = np.zeros((BATCH, FC_IN), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * FC_IN * 4)
    d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
    lib.gpu_memcpy_h2d(d_fc_in, h_pool.ctypes.data, BATCH * FC_IN * 4)
    
    d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
    
    # Loss
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    exp_out = np.exp(h_out_shifted)
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    
    # Backward
    labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
    labels_onehot[np.arange(BATCH), y] = 1.0
    d_loss = probs - labels_onehot
    
    # FC grad
    grad_fc_w = d_loss.T @ h_pool
    grad_fc_w_clipped = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
    d_fc_grad_w = lib.gpu_malloc(10 * FC_IN * 4)
    lib.gpu_memcpy_h2d(d_fc_grad_w, grad_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
    lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR), 10 * FC_IN)
    
    # FC -> Pool grad
    fc_w_reshaped = fc_w.reshape(10, FC_IN)
    grad_pool = d_loss @ fc_w_reshaped
    grad_pool_clipped = np.clip(grad_pool, -1.0, 1.0).astype(np.float32)
    d_pool_grad = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
    lib.gpu_memcpy_h2d(d_pool_grad, grad_pool_clipped.flatten().ctypes.data, OC * BATCH * 15 * 15 * 4)
    
    # Pool backward
    d_conv_grad = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
    lib.gpu_memset(d_conv_grad, 0, OC * BATCH * outH * outW * 4)
    lib.maxpool_backward_use_idx(d_pool_grad, d_max_idx, d_conv_grad, BATCH, OC, outH, outW)
    
    # Free memory
    lib.gpu_free(d_x)
    lib.gpu_free(d_col)
    lib.gpu_free(d_conv_raw)
    lib.gpu_free(d_conv)
    lib.gpu_free(d_pool)
    lib.gpu_free(d_max_idx)
    lib.gpu_free(d_fc_in)
    lib.gpu_free(d_fc_out)
    lib.gpu_free(d_fc_grad_w)
    lib.gpu_free(d_pool_grad)
    lib.gpu_free(d_conv_grad)
    
    if (batch_idx + 1) % 10 == 0:
        print(f"  Batch {batch_idx+1}: Loss={loss:.4f}")

lib.gpu_free(d_w_conv)
lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
print(f"Done! Time={time.time()-t0:.1f}s")
