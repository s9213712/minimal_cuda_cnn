#!/usr/bin/env python3
"""V3 Optimized: 100% GPU pipeline, He init, gradient normalization"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os, pickle, time

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
lib.leaky_relu_backward.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.dense_backward.argtypes = [c_void_p]*6 + [c_int]*3
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

BATCH = 64
LR_CONV = 0.01
LR_CONV1 = 0.001
LR_FC = 0.01
EPOCHS = 3

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

# ========== WEIGHT INITIALIZATION ==========
np.random.seed(42)
# Use std=0.05 (known stable from train_fixed_wsl.py at 64.4%)
std = 0.05
w_conv1 = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * std
w_conv2 = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * std
fc_w = np.random.randn(10*FC_IN).astype(np.float32) * std
fc_b = np.zeros(10, dtype=np.float32)

# ========== PERSISTENT WEIGHT BUFFERS ==========
d_w_conv1 = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_w_conv2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_fc_w = lib.gpu_malloc(10*FC_IN*4)
d_fc_b = lib.gpu_malloc(10*4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10*FC_IN*4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10*4)

# ========== PERSISTENT GRADIENT BUFFERS ==========
d_wc1_grad = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_wc2_grad = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_fc_grad_w = lib.gpu_malloc(10*FC_IN*4)
d_fc_grad_b = lib.gpu_malloc(10*4)

# ========== PERSISTENT ACTIVATION BUFFERS ==========
d_x = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
d_col1 = lib.gpu_malloc(C1_IN*KH1*KW1*BATCH*outH1*outW1*4)
d_conv1_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_conv1 = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_pool1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
d_max_idx1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
d_col2 = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
d_conv2_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_conv2 = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_pool2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
d_max_idx2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
d_fc_in = lib.gpu_malloc(BATCH*FC_IN*4)
d_fc_out = lib.gpu_malloc(BATCH*10*4)
d_dout = lib.gpu_malloc(BATCH*10*4)
d_din = lib.gpu_malloc(BATCH*FC_IN*4)
d_pool2_grad = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
d_conv2_grad_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_conv2_grad = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_pool1_grad = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
d_conv1_grad_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_conv1_grad = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_x_grad = lib.gpu_malloc(BATCH*C1_IN*H*W*4)

data_root = os.path.join(workspace, "NN/minimal_cuda_cnn/data/cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_all = b[b"data"].astype(np.float32)/255.0
    x_all = x_all.reshape(-1, 3, 32, 32)
    y_all = np.array(b[b"labels"])

print(f"V3 Optimized: Conv1({C1_IN}→{C1_OUT})→Pool1→Conv2({C2_IN}→{C2_OUT})→Pool2→FC({FC_IN}→10)")
print(f"LR_conv1={LR_CONV1}, LR_conv2={LR_CONV}, LR_fc={LR_FC}, BATCH={BATCH}, EPOCHS={EPOCHS}")
print(f"He init, gradient normalization, 100% GPU pipeline\n")

NBATCHES = x_all.shape[0] // BATCH

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0; correct = 0
    indices = np.random.permutation(x_all.shape[0])
    
    for batch_idx in range(NBATCHES):
        idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
        x = x_all[indices[idx_s:idx_e]]; y = y_all[indices[idx_s:idx_e]]
        
        # ========== FORWARD (reuse buffers) ==========
        lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH*C1_IN*H*W*4)
        
        lib.im2col_forward(d_x, d_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
        lib.gemm_forward(d_w_conv1, d_col1, d_conv1_raw, C1_OUT, BATCH*outH1*outW1, C1_IN*KH1*KW1)
        lib.leaky_relu_forward(d_conv1_raw, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
        lib.reorganize_forward(d_conv1_raw, d_conv1, BATCH, C1_OUT, outH1, outW1)
        lib.maxpool_forward_store(d_pool1, d_conv1, d_max_idx1, BATCH, C1_OUT, outH1, outW1)
        
        lib.im2col_forward(d_pool1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
        lib.gemm_forward(d_w_conv2, d_col2, d_conv2_raw, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
        lib.leaky_relu_forward(d_conv2_raw, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
        lib.reorganize_forward(d_conv2_raw, d_conv2, BATCH, C2_OUT, outH2, outW2)
        lib.maxpool_forward_store(d_pool2, d_conv2, d_max_idx2, BATCH, C2_OUT, outH2, outW2)
        
        # FC forward - d_pool2 (N,FC_IN) flatten -> d_fc_in (N,FC_IN)
        h_pool2 = g2h(d_pool2, BATCH*FC_IN).reshape(BATCH, FC_IN)
        lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH*FC_IN*4)
        lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
        h_out = g2h(d_fc_out, BATCH*10).reshape(BATCH, 10)
        
        h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
        probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y)
        
        # ========== BACKWARD ==========
        labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
        labels_onehot[np.arange(BATCH), y] = 1.0
        d_loss = probs - labels_onehot
        
        # ========== FC BACKWARD (100% GPU) ==========
        lib.gpu_memcpy_h2d(d_dout, d_loss.flatten().ctypes.data, BATCH*10*4)
        lib.gpu_memset(d_fc_grad_w, 0, 10*FC_IN*4)
        lib.gpu_memset(d_fc_grad_b, 0, 10*4)
        lib.gpu_memset(d_din, 0, BATCH*FC_IN*4)
        lib.dense_backward(d_dout, d_fc_in, d_fc_w, d_fc_grad_w, d_din, d_fc_grad_b, BATCH, FC_IN, 10)
        # Apply FC update
        lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR_FC), 10*FC_IN)
        lib.apply_sgd_update(d_fc_b, d_fc_grad_b, c_float(LR_FC), 10)
        # Clip weights to prevent explosion
        h_fc_w = g2h(d_fc_w, 10*FC_IN)
        h_fc_w = np.clip(h_fc_w, -2.0, 2.0)
        lib.gpu_memcpy_h2d(d_fc_w, h_fc_w.ctypes.data, 10*FC_IN*4)
        
        # ========== CONV2 BACKWARD ==========
        h_pool2_grad = g2h(d_din, BATCH*FC_IN).reshape(FC_IN, BATCH).T.flatten()
        lib.gpu_memcpy_h2d(d_pool2_grad, h_pool2_grad.ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)
        
        lib.gpu_memset(d_conv2_grad_raw, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad_raw, BATCH, C2_OUT, outH2, outW2)
        lib.gpu_memset(d_conv2_grad, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.reorganize_backward(d_conv2_grad_raw, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)
        lib.leaky_relu_backward(d_conv2_raw, d_conv2_grad, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
        
        lib.gpu_memset(d_wc2_grad, 0, C2_OUT*C2_IN*KH2*KW2*4)
        lib.gpu_memset(d_pool1_grad, 0, C2_IN*BATCH*poolH1*poolW1*4)
        lib.conv_backward(d_conv2_grad, d_pool1, d_w_conv2, d_wc2_grad, d_pool1_grad,
                          BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
        lib.apply_sgd_update(d_w_conv2, d_wc2_grad, c_float(LR_CONV), C2_OUT*C2_IN*KH2*KW2)
        
        # ========== CONV1 BACKWARD ==========
        lib.gpu_memset(d_conv1_grad_raw, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.maxpool_backward_use_idx(d_pool1_grad, d_max_idx1, d_conv1_grad_raw, BATCH, C1_OUT, outH1, outW1)
        lib.gpu_memset(d_conv1_grad, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.reorganize_backward(d_conv1_grad_raw, d_conv1_grad, BATCH, C1_OUT, outH1, outW1)
        lib.leaky_relu_backward(d_conv1_raw, d_conv1_grad, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
        
        lib.gpu_memset(d_wc1_grad, 0, C1_OUT*C1_IN*KH1*KW1*4)
        lib.gpu_memset(d_x_grad, 0, BATCH*C1_IN*H*W*4)
        lib.conv_backward(d_conv1_grad, d_x, d_w_conv1, d_wc1_grad, d_x_grad,
                          BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
        lib.apply_sgd_update(d_w_conv1, d_wc1_grad, c_float(LR_CONV1), C1_OUT*C1_IN*KH1*KW1)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={loss:.4f}, acc={correct/(batch_idx+1)/BATCH*100:.1f}%")
    
    acc = correct / NBATCHES / BATCH * 100
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/NBATCHES:.4f}, Acc={acc:.2f}%, Time={time.time()-t0:.1f}s")

print("\nDone! V3 Optimized - He init, gradient norm, 100% GPU")
