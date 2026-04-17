#!/usr/bin/env python3
"""Full training with COMPLETE backward chain - WSL"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os, pickle, time

workspace = "/mnt/c/Users/user/.openclaw/workspace"
so = os.path.join(workspace, "NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)

# Argtypes
lib.gpu_malloc.argtypes = [ctypes.c_size_t]; lib.gpu_malloc.restype = c_void_p
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
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.im2col_backward.argtypes = [c_void_p]*10  # (grad_col, grad_in, N, C, H, W, KH, KW, outH, outW)
lib.gemm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

BATCH = 64
LR_CONV = 0.005
LR_FC = 0.01
EPOCHS = 5

# Architecture
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

def check(name, arr):
    nan, inf = np.isnan(arr).sum(), np.isinf(arr).sum()
    ok = nan == 0 and inf == 0
    print(f"  {name}: {arr.min():.4f}/{arr.max():.4f} nan={nan} {'OK' if ok else 'BAD'}")
    return ok

# Init weights
np.random.seed(42)
std = 0.05
w_conv1 = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * std
w_conv2 = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * std
fc_w = np.random.randn(10*FC_IN).astype(np.float32) * std
fc_b = np.zeros(10, dtype=np.float32)

d_w_conv1 = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_w_conv2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_fc_w = lib.gpu_malloc(10*FC_IN*4)
d_fc_b = lib.gpu_malloc(10*4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10*FC_IN*4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10*4)

data_root = os.path.join(workspace, "NN/minimal_cuda_cnn/data/cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_all = b[b"data"].astype(np.float32)/255.0
    x_all = x_all.reshape(-1, 3, 32, 32)
    y_all = np.array(b[b"labels"])

print(f"Arch: Conv1({C1_IN}→{C1_OUT})→Pool({poolH1})→Conv2({C2_IN}→{C2_OUT})→Pool({poolH2})→FC({FC_IN}→10)")
print(f"LR_conv={LR_CONV}, LR_fc={LR_FC}, BATCH={BATCH}, EPOCHS={EPOCHS}")

NBATCHES = x_all.shape[0] // BATCH

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0; correct = 0
    indices = np.random.permutation(x_all.shape[0])
    
    for batch_idx in range(NBATCHES):
        idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
        x = x_all[indices[idx_s:idx_e]]; y = y_all[indices[idx_s:idx_e]]
        
        # === FORWARD ===
        d_x = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
        lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH*C1_IN*H*W*4)
        
        # Conv1
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
        
        # Conv2
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
        
        # FC
        h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, BATCH*FC_IN*4)
        d_fc_in = lib.gpu_malloc(BATCH*FC_IN*4)
        lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH*FC_IN*4)
        d_fc_out = lib.gpu_malloc(BATCH*10*4)
        lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
        h_out = np.zeros((BATCH, 10), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH*10*4)
        
        h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
        probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y)
        
        # === BACKWARD ===
        labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
        labels_onehot[np.arange(BATCH), y] = 1.0
        d_loss = probs - labels_onehot
        
        # FC grad
        grad_fc_w = d_loss.T @ h_pool2
        grad_pool2 = d_loss @ fc_w.reshape(10, FC_IN)
        
        # SGD FC
        d_fc_grad_w = lib.gpu_malloc(10*FC_IN*4)
        lib.gpu_memcpy_h2d(d_fc_grad_w, np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32).ctypes.data, 10*FC_IN*4)
        lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR_FC), 10*FC_IN)
        h_fc_w = g2h(d_fc_w, 10*FC_IN)
        h_fc_w = np.clip(h_fc_w, -1.0, 1.0).astype(np.float32)
        lib.gpu_memcpy_h2d(d_fc_w, h_fc_w.ctypes.data, 10*FC_IN*4)
        fc_w = h_fc_w.copy()
        
        # === CONV2 BACKWARD ===
        grad_pool2_clip = np.clip(grad_pool2, -1.0, 1.0).astype(np.float32)
        
        # pool2 backward: grad_pool2 → conv2_grad_raw
        d_pool2_grad = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
        lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2_clip.flatten().ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)
        d_conv2_grad_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_grad_raw, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad_raw, BATCH, C2_OUT, outH2, outW2)
        
        # reorganize_backward: (N,OC,H,W) → (OC,N,H,W)
        d_conv2_grad = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_grad, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.reorganize_backward(d_conv2_grad_raw, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)
        
        # leaky_relu backward
        lib.leaky_relu_backward(d_conv2_raw, d_conv2_grad, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
        
        # conv2 backward: computes w_conv2 grad AND col2 grad
        d_w_conv2_grad = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
        lib.gpu_memset(d_w_conv2_grad, 0, C2_OUT*C2_IN*KH2*KW2*4)
        lib.conv_backward(d_conv2_grad, d_pool1, d_w_conv2, d_w_conv2_grad,
                          BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
        lib.apply_sgd_update(d_w_conv2, d_w_conv2_grad, c_float(LR_CONV), C2_OUT*C2_IN*KH2*KW2)
        h_w_conv2 = g2h(d_w_conv2, C2_OUT*C2_IN*KH2*KW2)
        h_w_conv2 = np.clip(h_w_conv2, -2.0, 2.0).astype(np.float32)
        lib.gpu_memcpy_h2d(d_w_conv2, h_w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
        w_conv2 = h_w_conv2.copy()
        
        # gemm_backward: col2_grad → conv2_raw_grad
        # Forward: conv2_raw = gemm(w_conv2, col2) → d_col2 = w_conv2^T @ d_conv2_grad
        # gemm_backward(grad_out, A, B, grad_A, grad_B, M, N, K)
        # Here: d_col2_grad = w_conv2^T @ d_conv2_grad  (K×N)
        # K = C2_IN*KH2*KW2 = 288, N = BATCH*outH2*outW2, M = C2_OUT = 64
        d_col2_grad = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
        d_conv2_raw_grad = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)  # unused output
        lib.gpu_memset(d_col2_grad, 0, C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_raw_grad, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.gemm_backward(d_conv2_grad, d_w_conv2, d_col2, d_conv2_raw_grad, d_col2_grad,
                          C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
        
        # im2col_backward: col2_grad → pool1_grad
        # Forward: col2 = im2col(pool1) → pool1_grad = im2col_backward(col2_grad)
        d_pool1_grad = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
        lib.gpu_memset(d_pool1_grad, 0, C2_IN*BATCH*poolH1*poolW1*4)
        lib.im2col_backward(d_col2_grad, d_pool1_grad, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
        
        # === CONV1 BACKWARD ===
        # pool1 backward: pool1_grad → conv1_grad_raw
        d_conv1_grad_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.gpu_memset(d_conv1_grad_raw, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.maxpool_backward_use_idx(d_pool1_grad, d_max_idx1, d_conv1_grad_raw, BATCH, C1_OUT, outH1, outW1)
        
        # reorganize_backward: (N,OC,H,W) → (OC,N,H,W)
        d_conv1_grad = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.gpu_memset(d_conv1_grad, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.reorganize_backward(d_conv1_grad_raw, d_conv1_grad, BATCH, C1_OUT, outH1, outW1)
        
        # leaky_relu backward
        lib.leaky_relu_backward(d_conv1_raw, d_conv1_grad, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
        
        # conv1 backward: w_conv1 grad
        d_w_conv1_grad = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
        lib.gpu_memset(d_w_conv1_grad, 0, C1_OUT*C1_IN*KH1*KW1*4)
        lib.conv_backward(d_conv1_grad, d_x, d_w_conv1, d_w_conv1_grad,
                          BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
        lib.apply_sgd_update(d_w_conv1, d_w_conv1_grad, c_float(LR_CONV), C1_OUT*C1_IN*KH1*KW1)
        h_w_conv1 = g2h(d_w_conv1, C1_OUT*C1_IN*KH1*KW1)
        h_w_conv1 = np.clip(h_w_conv1, -2.0, 2.0).astype(np.float32)
        lib.gpu_memcpy_h2d(d_w_conv1, h_w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
        w_conv1 = h_w_conv1.copy()
        
        # FREE
        lib.gpu_free(d_x); lib.gpu_free(d_col1); lib.gpu_free(d_conv1_raw); lib.gpu_free(d_conv1)
        lib.gpu_free(d_pool1); lib.gpu_free(d_max_idx1); lib.gpu_free(d_col2); lib.gpu_free(d_conv2_raw)
        lib.gpu_free(d_conv2); lib.gpu_free(d_pool2); lib.gpu_free(d_max_idx2); lib.gpu_free(d_fc_in)
        lib.gpu_free(d_fc_out); lib.gpu_free(d_fc_grad_w); lib.gpu_free(d_pool2_grad)
        lib.gpu_free(d_conv2_grad_raw); lib.gpu_free(d_conv2_grad); lib.gpu_free(d_w_conv2_grad)
        lib.gpu_free(d_col2_grad); lib.gpu_free(d_conv2_raw_grad); lib.gpu_free(d_pool1_grad)
        lib.gpu_free(d_conv1_grad_raw); lib.gpu_free(d_conv1_grad); lib.gpu_free(d_w_conv1_grad)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={loss:.4f}, acc={correct/(batch_idx+1)/BATCH*100:.1f}%")
    
    acc = correct / NBATCHES / BATCH * 100
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/NBATCHES:.4f}, Acc={acc:.2f}%, Time={time.time()-t0:.1f}s")

lib.gpu_free(d_w_conv1); lib.gpu_free(d_w_conv2); lib.gpu_free(d_fc_w); lib.gpu_free(d_fc_b)
print("Done!")
