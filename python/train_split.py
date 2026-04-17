#!/usr/bin/env python3
"""Full training with FIXED conv_backward - WSL
Train/Val/Test split: 40000 train, 5000 val, 10000 test
BATCH=32 + HFlip augmentation
"""
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
lib.leaky_relu_forward_nchw.argtypes = [c_void_p, c_float, c_int]
lib.leaky_relu_backward_nchw.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.nchw_to_cnhw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.cnhw_to_nchw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

BATCH = 64  # BATCH=32 too small, use 64
LR_CONV = 0.005
LR_CONV1 = 0.001
LR_FC = 0.005
EPOCHS = 20

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

data_root = os.path.join(workspace, "NN/minimal_cuda_cnn/data/cifar-10-batches-py")

# Load ALL 5 training batches
print("Loading 5 training batches...")
x_train_all = []
y_train_all = []
for i in range(1, 6):
    with open(os.path.join(data_root, f"data_batch_{i}"), "rb") as f:
        b = pickle.load(f, encoding="bytes")
        x_train_all.append(b[b"data"].astype(np.float32)/255.0)
        y_train_all.append(np.array(b[b"labels"]))
x_train_all = np.concatenate(x_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)
print(f"Total training samples: {x_train_all.shape[0]}")

# Reshape to NCHW
x_train_all = x_train_all.reshape(-1, 3, 32, 32)

# Load test batch
with open(os.path.join(data_root, "test_batch"), "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_test = b[b"data"].astype(np.float32)/255.0
    y_test = np.array(b[b"labels"])
x_test = x_test.reshape(-1, 3, 32, 32)
print(f"Test samples: {x_test.shape[0]}")

# Only use data_batch_1 (10000 samples) for faster training
print("Loading data_batch_1 only...")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_train_all = b[b"data"].astype(np.float32)/255.0
    y_train_all = np.array(b[b"labels"])
x_train_all = x_train_all.reshape(-1, 3, 32, 32)
print(f"Training samples: {x_train_all.shape[0]}")

# Split: 8000 train, 2000 val
n_train = 8000
n_val = 2000

indices = np.random.permutation(x_train_all.shape[0])
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

x_train = x_train_all[train_idx]
y_train = y_train_all[train_idx]
x_val = x_train_all[val_idx]
y_val = y_train_all[val_idx]
x_test_split = x_train_all[test_idx]
y_test_split = y_train_all[test_idx]

print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test(split): {x_test_split.shape[0]}")

# Also use official test batch as final test
x_test_final = x_test
y_test_final = y_test
print(f"Test(official): {x_test_final.shape[0]}")

# Initialize weights
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

print(f"Arch: Conv1({C1_IN}→{C1_OUT})→Pool1→Conv2({C2_IN}→{C2_OUT})→Pool2→FC({FC_IN}→10)")
print(f"LR_conv1={LR_CONV1}, LR_conv2={LR_CONV}, LR_fc={LR_FC}, BATCH={BATCH}, EPOCHS={EPOCHS}")
print(f"Train/Val/Test: {x_train.shape[0]}/{x_val.shape[0]}/{x_test_final.shape[0]}")
print()

def forward_batch(x, lib, d_w_conv1, d_w_conv2, d_fc_w, d_fc_b):
    """Run forward pass for a batch"""
    B = x.shape[0]
    d_x = lib.gpu_malloc(B*C1_IN*H*W*4)
    lib.gpu_memcpy_h2d(d_x, x.ctypes.data, B*C1_IN*H*W*4)
    
    d_col1 = lib.gpu_malloc(C1_IN*KH1*KW1*B*outH1*outW1*4)
    d_conv1_raw = lib.gpu_malloc(C1_OUT*B*outH1*outW1*4)
    lib.im2col_forward(d_x, d_col1, B, C1_IN, H, W, KH1, KW1, outH1, outW1)
    lib.gemm_forward(d_w_conv1, d_col1, d_conv1_raw, C1_OUT, B*outH1*outW1, C1_IN*KH1*KW1)
    lib.leaky_relu_forward(d_conv1_raw, c_float(0.1), C1_OUT*B*outH1*outW1)
    d_conv1 = lib.gpu_malloc(C1_OUT*B*outH1*outW1*4)
    lib.reorganize_forward(d_conv1_raw, d_conv1, B, C1_OUT, outH1, outW1)
    d_pool1 = lib.gpu_malloc(C1_OUT*B*poolH1*poolW1*4)
    d_max_idx1 = lib.gpu_malloc(C1_OUT*B*poolH1*poolW1*4)
    lib.maxpool_forward_store(d_pool1, d_conv1, d_max_idx1, B, C1_OUT, outH1, outW1)
    
    d_col2 = lib.gpu_malloc(C2_IN*KH2*KW2*B*outH2*outW2*4)
    d_conv2_raw = lib.gpu_malloc(C2_OUT*B*outH2*outW2*4)
    lib.im2col_forward(d_pool1, d_col2, B, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
    lib.gemm_forward(d_w_conv2, d_col2, d_conv2_raw, C2_OUT, B*outH2*outW2, C2_IN*KH2*KW2)
    lib.leaky_relu_forward(d_conv2_raw, c_float(0.1), C2_OUT*B*outH2*outW2)
    d_conv2 = lib.gpu_malloc(C2_OUT*B*outH2*outW2*4)
    lib.reorganize_forward(d_conv2_raw, d_conv2, B, C2_OUT, outH2, outW2)
    d_pool2 = lib.gpu_malloc(C2_OUT*B*poolH2*poolW2*4)
    d_max_idx2 = lib.gpu_malloc(C2_OUT*B*poolH2*poolW2*4)
    lib.maxpool_forward_store(d_pool2, d_conv2, d_max_idx2, B, C2_OUT, outH2, outW2)
    
    h_pool2 = np.zeros((B, FC_IN), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, B*FC_IN*4)
    d_fc_in = lib.gpu_malloc(B*FC_IN*4)
    lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, B*FC_IN*4)
    d_fc_out = lib.gpu_malloc(B*10*4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, B, FC_IN, 10)
    h_out = np.zeros((B, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, B*10*4)
    
    # Free batch memory
    for ptr in [d_x, d_col1, d_conv1_raw, d_conv1, d_pool1, d_max_idx1, d_col2, d_conv2_raw, d_conv2, d_pool2, d_max_idx2, d_fc_in, d_fc_out]:
        lib.gpu_free(ptr)
    
    return h_out

def evaluate(x, y, lib, d_w_conv1, d_w_conv2, d_fc_w, d_fc_b, batch_size=BATCH, max_batches=50):
    """Evaluate accuracy on a dataset (limited batches)"""
    correct = 0; total = 0
    nbatches = min(x.shape[0] // batch_size, max_batches)
    for i in range(nbatches):
        idx_s = i * batch_size; idx_e = idx_s + batch_size
        h_out = forward_batch(x[idx_s:idx_e], lib, d_w_conv1, d_w_conv2, d_fc_w, d_fc_b)
        preds = np.argmax(h_out, axis=1)
        correct += np.sum(preds == y[idx_s:idx_e])
        total += batch_size
    return correct / total * 100

NBATCHES = x_train.shape[0] // BATCH

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0; correct = 0
    indices = np.random.permutation(x_train.shape[0])
    
    for batch_idx in range(NBATCHES):
        idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
        x = x_train[indices[idx_s:idx_e]]; y = y_train[indices[idx_s:idx_e]]
        
        # === DATA AUGMENTATION: Horizontal Flip ===
        if np.random.rand() > 0.5:
            x = x.copy()
            x = x[:, :, :, ::-1].copy()
        
        # === FORWARD ===
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
        
        h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
        probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y)
        
        # === BACKWARD ===
        labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
        labels_onehot[np.arange(BATCH), y] = 1.0
        d_loss = probs - labels_onehot
        
        # FC SGD — FIX: divide grad by BATCH for stability, compute bias grad, loosen clip
        grad_fc_w = (d_loss.T @ h_pool2) / BATCH      # was: no division → 64x gradient explosion
        grad_fc_b = d_loss.sum(axis=0) / BATCH         # NEW: compute bias gradient (was completely missing!)
        grad_pool2 = (d_loss @ fc_w.reshape(10, FC_IN)) / BATCH  # divide too for consistency

        # --- SGD UPDATE ---
        d_fc_grad_w = lib.gpu_malloc(10*FC_IN*4)
        lib.gpu_memcpy_h2d(d_fc_grad_w, np.clip(grad_fc_w, -1.0, 1.0).astype(np.float32).ctypes.data, 10*FC_IN*4)
        lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR_FC), 10*FC_IN)
        d_fc_grad_b = lib.gpu_malloc(10*4)
        lib.gpu_memcpy_h2d(d_fc_grad_b, np.clip(grad_fc_b, -5.0, 5.0).astype(np.float32).ctypes.data, 10*4)
        lib.apply_sgd_update(d_fc_b, d_fc_grad_b, c_float(LR_FC), 10)
        lib.gpu_free(d_fc_grad_w)
        lib.gpu_free(d_fc_grad_b)
        h_fc_w = g2h(d_fc_w, 10*FC_IN).astype(np.float32)
        h_fc_b = g2h(d_fc_b, 10).astype(np.float32)
        lib.gpu_memcpy_h2d(d_fc_w, h_fc_w.ctypes.data, 10*FC_IN*4)
        lib.gpu_memcpy_h2d(d_fc_b, h_fc_b.ctypes.data, 10*4)
        fc_w = h_fc_w.copy()
        fc_b = h_fc_b.copy()
        
        # === CONV2 BACKWARD ===
        # Forward flow: x(NCHW) → conv1_raw → leaky_relu → reorganize → pool1 → conv2_raw → leaky_relu → reorganize → pool2
        # Backward flow: pool2_grad → maxpool_backward → conv2_grad_raw → leaky_relu_backward → reorganize_backward → conv2_grad → conv_backward → pool1_grad
        #
        # Layout summary:
        #   grad_pool2: (BATCH, FC_IN) in NCHW row-major = (BATCH, C2_OUT*poolH2*poolW2)
        #   d_pool2_grad: (C2_OUT, BATCH, poolH2, poolW2) CNHW — allocate as CNHW, fill with NCHW→CNHW conversion
        #   d_conv2_grad_raw: (C2_OUT, BATCH, outH2, outW2) CNHW — for maxpool_backward output
        #   d_conv2_raw: (C2_OUT, BATCH, outH2, outW2) CNHW — forward leaky_relu output
        #
        # Step 1: maxpool_backward_use_idx — CNHW grad_out + CNHW indices → CNHW grad_input
        #   grad_pool2 (BATCH, C2_OUT, poolH2, poolW2) NCHW → convert to d_pool2_grad (C2_OUT, BATCH, poolH2, poolW2) CNHW
        #   maxpool stores indices pointing into conv2 output space (outH2×outW2)
        #   d_conv2_grad_raw must be (C2_OUT, BATCH, outH2, outW2) CNHW
        grad_pool2_clip = np.clip(grad_pool2, -1.0, 1.0).astype(np.float32)
        # grad_pool2 is (BATCH, C2_OUT, poolH2, poolW2) NCHW
        # d_pool2_grad is (C2_OUT, BATCH, poolH2, poolW2) CNHW
        grad_pool2_reshaped = grad_pool2_clip.reshape(BATCH, C2_OUT, poolH2, poolW2)  # NCHW
        grad_pool2_cnhw = np.transpose(grad_pool2_reshaped, (1, 0, 2, 3)).flatten()  # CNHW: (C,N,H,W) → flatten
        d_pool2_grad = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
        lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2_cnhw.ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)

        d_conv2_grad_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_grad_raw, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad_raw, BATCH, C2_OUT, outH2, outW2)

        # Step 2: leaky_relu_backward on CNHW data
        # d_conv2_raw is CNHW, d_conv2_grad_raw is CNHW — both use same CNHW indexing
        lib.leaky_relu_backward(d_conv2_raw, d_conv2_grad_raw, c_float(0.1), C2_OUT*BATCH*outH2*outW2)

        # Step 3: reorganize_backward — d_conv2_grad_raw is CNHW, reorganize_backward expects NCHW
        # reorganize_backward: NCHW grad → CNHW grad
        # d_conv2_grad_nchw = NCHW version of CNHW d_conv2_grad_raw
        d_conv2_grad_nchw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.cnhw_to_nchw(d_conv2_grad_raw, d_conv2_grad_nchw, BATCH, C2_OUT, outH2, outW2)
        d_conv2_grad = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_grad, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.reorganize_backward(d_conv2_grad_nchw, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)

        # Step 4: conv_backward — conv_backward expects NCHW for both grad_out and input
        # d_conv2_grad is CNHW, convert to NCHW for conv_backward
        d_conv2_grad_for_conv = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.cnhw_to_nchw(d_conv2_grad, d_conv2_grad_for_conv, BATCH, C2_OUT, outH2, outW2)
        # d_pool1 is CNHW, convert to NCHW for conv_backward
        d_pool1_nchw = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
        lib.cnhw_to_nchw(d_pool1, d_pool1_nchw, BATCH, C2_IN, poolH1, poolW1)

        d_w_conv2_grad = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
        d_pool1_grad = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
        lib.gpu_memset(d_w_conv2_grad, 0, C2_OUT*C2_IN*KH2*KW2*4)
        lib.gpu_memset(d_pool1_grad, 0, C2_IN*BATCH*poolH1*poolW1*4)
        lib.conv_backward(d_conv2_grad_for_conv, d_pool1_nchw, d_w_conv2, d_w_conv2_grad, d_pool1_grad,
                          BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
        # Convert pool1 grad to CNHW for pool1 backward
        d_pool1_grad_cnhw = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
        lib.nchw_to_cnhw(d_pool1_grad, d_pool1_grad_cnhw, BATCH, C2_IN, poolH1, poolW1)
        # FIX: divide conv weight grad by BATCH (atomicAdd sums all batch elements, so mean = sum/BATCH)
        h_w_conv2_grad = g2h(d_w_conv2_grad, C2_OUT*C2_IN*KH2*KW2).reshape(-1) / BATCH
        lib.gpu_memcpy_h2d(d_w_conv2_grad, np.clip(h_w_conv2_grad, -1.0, 1.0).astype(np.float32).ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
        lib.apply_sgd_update(d_w_conv2, d_w_conv2_grad, c_float(LR_CONV), C2_OUT*C2_IN*KH2*KW2)
        h_w_conv2 = g2h(d_w_conv2, C2_OUT*C2_IN*KH2*KW2).astype(np.float32)
        lib.gpu_memcpy_h2d(d_w_conv2, h_w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
        w_conv2 = h_w_conv2.copy()

        # FREE intermediate conv2 backward buffers
        lib.gpu_free(d_conv2_grad_nchw)
        lib.gpu_free(d_conv2_grad_for_conv)
        lib.gpu_free(d_pool1_nchw)

        # === CONV1 BACKWARD ===
        # Forward flow: x(NCHW) → conv1_raw → leaky_relu → reorganize → pool1
        # Backward flow: pool1_grad → maxpool_backward → conv1_grad_raw → leaky_relu_backward → reorganize_backward → conv1_grad → conv_backward
        #
        # Step 1: maxpool_backward_use_idx — pool1_grad(CNHW) + max_idx1 → conv1_grad_raw(CNHW)
        d_conv1_grad_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.gpu_memset(d_conv1_grad_raw, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.maxpool_backward_use_idx(d_pool1_grad_cnhw, d_max_idx1, d_conv1_grad_raw, BATCH, C1_OUT, outH1, outW1)

        # Step 2: leaky_relu_backward on CNHW data
        # d_conv1_raw is CNHW, d_conv1_grad_raw is CNHW
        lib.leaky_relu_backward(d_conv1_raw, d_conv1_grad_raw, c_float(0.1), C1_OUT*BATCH*outH1*outW1)

        # Step 3: reorganize_backward — CNHW grad
        d_conv1_grad_nchw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.cnhw_to_nchw(d_conv1_grad_raw, d_conv1_grad_nchw, BATCH, C1_OUT, outH1, outW1)
        d_conv1_grad = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.gpu_memset(d_conv1_grad, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.reorganize_backward(d_conv1_grad_nchw, d_conv1_grad, BATCH, C1_OUT, outH1, outW1)

        # Step 4: conv_backward — NCHW grad_out, NCHW input, NCHW weights
        d_conv1_grad_for_conv = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.cnhw_to_nchw(d_conv1_grad, d_conv1_grad_for_conv, BATCH, C1_OUT, outH1, outW1)
        # d_x is NCHW from input, no conversion needed
        d_w_conv1_grad = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
        d_x_grad = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
        lib.gpu_memset(d_w_conv1_grad, 0, C1_OUT*C1_IN*KH1*KW1*4)
        lib.gpu_memset(d_x_grad, 0, BATCH*C1_IN*H*W*4)
        lib.conv_backward(d_conv1_grad_for_conv, d_x, d_w_conv1, d_w_conv1_grad, d_x_grad,
                          BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
        # FIX: divide conv weight grad by BATCH (atomicAdd sums all batch elements, so mean = sum/BATCH)
        h_w_conv1_grad = g2h(d_w_conv1_grad, C1_OUT*C1_IN*KH1*KW1).reshape(-1) / BATCH
        lib.gpu_memcpy_h2d(d_w_conv1_grad, np.clip(h_w_conv1_grad, -1.0, 1.0).astype(np.float32).ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
        lib.apply_sgd_update(d_w_conv1, d_w_conv1_grad, c_float(LR_CONV1), C1_OUT*C1_IN*KH1*KW1)
        h_w_conv1 = g2h(d_w_conv1, C1_OUT*C1_IN*KH1*KW1).astype(np.float32)
        lib.gpu_memcpy_h2d(d_w_conv1, h_w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
        w_conv1 = h_w_conv1.copy()

        # FREE conv1 backward intermediates
        lib.gpu_free(d_conv1_grad_nchw)
        lib.gpu_free(d_conv1_grad_for_conv)
        
        # FREE
        lib.gpu_free(d_x); lib.gpu_free(d_col1); lib.gpu_free(d_conv1_raw); lib.gpu_free(d_conv1)
        lib.gpu_free(d_pool1); lib.gpu_free(d_max_idx1); lib.gpu_free(d_col2); lib.gpu_free(d_conv2_raw)
        lib.gpu_free(d_conv2); lib.gpu_free(d_pool2); lib.gpu_free(d_max_idx2); lib.gpu_free(d_fc_in)
        lib.gpu_free(d_fc_out); lib.gpu_free(d_fc_grad_w); lib.gpu_free(d_fc_grad_b)
        # Conv2 backward frees
        lib.gpu_free(d_pool2_grad); lib.gpu_free(d_conv2_grad); lib.gpu_free(d_conv2_grad_raw)
        lib.gpu_free(d_w_conv2_grad); lib.gpu_free(d_pool1_grad); lib.gpu_free(d_pool1_grad_cnhw)
        # Conv1 backward frees
        lib.gpu_free(d_conv1_grad); lib.gpu_free(d_conv1_grad_raw)
        lib.gpu_free(d_w_conv1_grad); lib.gpu_free(d_x_grad)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={loss:.4f}, acc={correct/(batch_idx+1)/BATCH*100:.1f}%")
    
    train_acc = correct / NBATCHES / BATCH * 100
    
    # Evaluate on validation set
    val_acc = evaluate(x_val, y_val, lib, d_w_conv1, d_w_conv2, d_fc_w, d_fc_b)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/NBATCHES:.4f}, Train={train_acc:.2f}%, Val={val_acc:.2f}%, Time={time.time()-t0:.1f}s")

# Final test on official test batch
print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
test_acc = evaluate(x_test_final, y_test_final, lib, d_w_conv1, d_w_conv2, d_fc_w, d_fc_b)
print(f"Test Accuracy: {test_acc:.2f}%")

lib.gpu_free(d_w_conv1); lib.gpu_free(d_w_conv2); lib.gpu_free(d_fc_w); lib.gpu_free(d_fc_b)
print("\nDone!")
