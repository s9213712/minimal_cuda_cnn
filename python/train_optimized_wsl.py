#!/usr/bin/env python3
"""Memory-optimized training - reuse buffers, no per-batch alloc/free"""
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

# Pre-allocate persistent GPU buffers
d_w_conv1 = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_w_conv2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_fc_w = lib.gpu_malloc(10*FC_IN*4)
d_fc_b = lib.gpu_malloc(10*4)

# Per-batch buffers (reused each batch)
d_x         = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
d_col1      = lib.gpu_malloc(C1_IN*KH1*KW1*BATCH*outH1*outW1*4)
d_conv1_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_conv1     = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_pool1     = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
d_max_idx1  = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
d_col2      = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
d_conv2_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_conv2     = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_pool2     = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
d_max_idx2  = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
d_fc_in     = lib.gpu_malloc(BATCH*FC_IN*4)
d_fc_out    = lib.gpu_malloc(BATCH*10*4)
d_pool2_grad    = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
d_conv2g_raw    = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_conv2_grad    = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
d_wc2_grad      = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
d_pool1_grad    = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
d_conv1g_raw    = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_conv1_grad    = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
d_wc1_grad      = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
d_x_grad        = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
d_fc_grad_w     = lib.gpu_malloc(10*FC_IN*4)

# Init weights
np.random.seed(42)
std = 0.05
w_conv1 = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * std
w_conv2 = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * std
fc_w = np.random.randn(10*FC_IN).astype(np.float32) * std
fc_b = np.zeros(10, dtype=np.float32)
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

print(f"Pre-allocated {24} persistent buffers")
print(f"LR_conv1={LR_CONV1}, LR_conv2={LR_CONV}, LR_fc={LR_FC}, BATCH={BATCH}")
print()

def free_all():
    for ptr in [d_w_conv1, d_w_conv2, d_fc_w, d_fc_b, d_x, d_col1, d_conv1_raw, d_conv1,
                d_pool1, d_max_idx1, d_col2, d_conv2_raw, d_conv2, d_pool2, d_max_idx2,
                d_fc_in, d_fc_out, d_pool2_grad, d_conv2g_raw, d_conv2_grad, d_wc2_grad,
                d_pool1_grad, d_conv1g_raw, d_conv1_grad, d_wc1_grad, d_x_grad, d_fc_grad_w]:
        lib.gpu_free(ptr)

TOTAL_BATCHES = x_all.shape[0] // BATCH
TARGET_BATCHES = 1560  # 10 epochs worth

print(f"Starting training: {TOTAL_BATCHES} batches/epoch, target {TARGET_BATCHES} total")
print(f"Time estimate: {64 * TARGET_BATCHES / 60:.0f} min at 64s/epoch")
print()

batch_count = 0
total_loss = 0.0; correct = 0
t0 = time.time()
indices = np.random.permutation(x_all.shape[0])

while batch_count < TARGET_BATCHES:
    epoch_batch = batch_count % TOTAL_BATCHES
    
    if epoch_batch == 0 and batch_count > 0:
        acc = correct / TOTAL_BATCHES / BATCH * 100
        elapsed = time.time() - t0
        epoch_num = batch_count // TOTAL_BATCHES
        print(f"Epoch {epoch_num}: Acc={acc:.2f}%, Avg={total_loss/TOTAL_BATCHES:.4f}, Time={elapsed:.0f}s")
        total_loss = 0.0; correct = 0
        indices = np.random.permutation(x_all.shape[0])
    
    idx_s = epoch_batch * BATCH; idx_e = idx_s + BATCH
    x = x_all[indices[idx_s:idx_e]]; y = y_all[indices[idx_s:idx_e]]
    
    # === FORWARD ===
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
    
    h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, BATCH*FC_IN*4)
    lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH*FC_IN*4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH*10*4)
    
    h_out_s = h_out - h_out.max(axis=1, keepdims=True)
    probs = np.exp(h_out_s) / np.exp(h_out_s).sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    total_loss += loss
    correct += np.sum(np.argmax(probs, axis=1) == y)
    
    # === BACKWARD ===
    labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
    labels_onehot[np.arange(BATCH), y] = 1.0
    d_loss = probs - labels_onehot
    
    # FC
    grad_fc_w = d_loss.T @ h_pool2
    grad_pool2 = d_loss @ fc_w.reshape(10, FC_IN)
    lib.gpu_memcpy_h2d(d_fc_grad_w, np.clip(grad_fc_w, -1.0, 1.0).flatten().ctypes.data, 10*FC_IN*4)
    lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR_FC), 10*FC_IN)
    h_fc_w = np.clip(g2h(d_fc_w, 10*FC_IN), -1.0, 1.0)
    lib.gpu_memcpy_h2d(d_fc_w, h_fc_w.ctypes.data, 10*FC_IN*4)
    fc_w = h_fc_w.copy()
    
    # Conv2 backward
    lib.gpu_memcpy_h2d(d_pool2_grad, np.clip(grad_pool2, -1.0, 1.0).flatten().ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)
    lib.gpu_memset(d_conv2g_raw, 0, C2_OUT*BATCH*outH2*outW2*4)
    lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2g_raw, BATCH, C2_OUT, outH2, outW2)
    lib.gpu_memset(d_conv2_grad, 0, C2_OUT*BATCH*outH2*outW2*4)
    lib.reorganize_backward(d_conv2g_raw, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)
    lib.leaky_relu_backward(d_conv2_raw, d_conv2_grad, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
    lib.gpu_memset(d_wc2_grad, 0, C2_OUT*C2_IN*KH2*KW2*4)
    lib.conv_backward(d_conv2_grad, d_pool1, d_w_conv2, d_wc2_grad, d_pool1_grad,
                      BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
    lib.apply_sgd_update(d_w_conv2, d_wc2_grad, c_float(LR_CONV), C2_OUT*C2_IN*KH2*KW2)
    h_wc2 = np.clip(g2h(d_w_conv2, C2_OUT*C2_IN*KH2*KW2), -2.0, 2.0)
    lib.gpu_memcpy_h2d(d_w_conv2, h_wc2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
    w_conv2 = h_wc2.copy()
    
    # Conv1 backward
    lib.gpu_memset(d_conv1g_raw, 0, C1_OUT*BATCH*outH1*outW1*4)
    lib.maxpool_backward_use_idx(d_pool1_grad, d_max_idx1, d_conv1g_raw, BATCH, C1_OUT, outH1, outW1)
    lib.gpu_memset(d_conv1_grad, 0, C1_OUT*BATCH*outH1*outW1*4)
    lib.reorganize_backward(d_conv1g_raw, d_conv1_grad, BATCH, C1_OUT, outH1, outW1)
    lib.leaky_relu_backward(d_conv1_raw, d_conv1_grad, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
    lib.gpu_memset(d_wc1_grad, 0, C1_OUT*C1_IN*KH1*KW1*4)
    lib.conv_backward(d_conv1_grad, d_x, d_w_conv1, d_wc1_grad, d_x_grad,
                      BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
    lib.apply_sgd_update(d_w_conv1, d_wc1_grad, c_float(LR_CONV1), C1_OUT*C1_IN*KH1*KW1)
    h_wc1 = np.clip(g2h(d_w_conv1, C1_OUT*C1_IN*KH1*KW1), -2.0, 2.0)
    lib.gpu_memcpy_h2d(d_w_conv1, h_wc1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
    w_conv1 = h_wc1.copy()
    
    batch_count += 1
    if batch_count % 50 == 0:
        print(f"  Batch {batch_count}/{TARGET_BATCHES}: loss={loss:.4f}, acc={correct/batch_count/BATCH*100:.1f}%")

free_all()
print(f"\nFinal: {batch_count} batches completed")
print("Done!")
