#!/usr/bin/env python3
"""Evaluate trained model on CIFAR-10 test set"""
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
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH = 64
C1_IN, C1_OUT = 3, 32; H, W = 32, 32; KH1, KW1 = 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1
poolH1, poolW1 = outH1 // 2, outW1 // 2
C2_IN, C2_OUT = 32, 64; KH2, KW2 = 3, 3
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2
FC_IN = C2_OUT * poolH2 * poolW2

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

# Load trained weights
data_root = os.path.join(workspace, "NN/minimal_cuda_cnn/data/cifar-10-batches-py")
w_conv1 = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * 0.05
w_conv2 = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * 0.05
fc_w = np.random.randn(10*FC_IN).astype(np.float32) * 0.05
fc_b = np.zeros(10, dtype=np.float32)

# Allocate GPU memory
d_w_conv1 = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_w_conv2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_fc_w = lib.gpu_malloc(10*FC_IN*4)
d_fc_b = lib.gpu_malloc(10*4)

# Load test batch
print("Loading test batch...")
with open(os.path.join(data_root, "test_batch"), "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_test = b[b"data"].astype(np.float32)/255.0
    x_test = x_test.reshape(-1, 3, 32, 32)
    y_test = np.array(b[b"labels"])

print(f"Test set: {x_test.shape[0]} samples")

# Copy weights to GPU
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10*FC_IN*4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10*4)

# Forward pass on test set
NBATCHES = x_test.shape[0] // BATCH
correct = 0; total = 0

for batch_idx in range(NBATCHES):
    idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
    x = x_test[idx_s:idx_e]; y = y_test[idx_s:idx_e]
    
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
    
    h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, BATCH*FC_IN*4)
    d_fc_in = lib.gpu_malloc(BATCH*FC_IN*4)
    lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH*FC_IN*4)
    d_fc_out = lib.gpu_malloc(BATCH*10*4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH*10*4)
    
    preds = np.argmax(h_out, axis=1)
    correct += np.sum(preds == y)
    total += BATCH
    
    if batch_idx % 50 == 0:
        print(f"Batch {batch_idx}/{NBATCHES}: acc={correct/total*100:.2f}%")
    
    # Free batch memory
    for ptr in [d_x, d_col1, d_conv1_raw, d_conv1, d_pool1, d_max_idx1, d_col2, d_conv2_raw, d_conv2, d_pool2, d_max_idx2, d_fc_in, d_fc_out]:
        lib.gpu_free(ptr)

print(f"\n=== TEST ACCURACY: {correct/total*100:.2f}% ===")
print(f"Correct: {correct}/{total}")
