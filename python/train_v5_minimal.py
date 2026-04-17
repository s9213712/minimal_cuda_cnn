#!/usr/bin/env python3
"""V5: Minimal BATCH=16, Early Stopping, Data Aug - WSL"""
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

BATCH = 16  # Very small batch for limited VRAM
LR_CONV = 0.005
LR_CONV1 = 0.001
LR_FC = 0.005
EPOCHS = 100
PATIENCE = 5
WD = 1e-4
DROPOUT = 0.3

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

def augment(x_batch):
    x_aug = x_batch.copy()
    for i in range(x_aug.shape[0]):
        if np.random.rand() > 0.5:
            x_aug[i] = x_aug[i, :, :, ::-1].copy()
        pad = 4
        img = x_aug[i]
        padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
        sh = np.random.randint(0, 2 * pad + 1)
        sw = np.random.randint(0, 2 * pad + 1)
        x_aug[i] = padded[:, sh:sh + H, sw:sw + W]
    return x_aug

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
x_train, y_train = [], []
for i in range(1, 6):
    with open(os.path.join(data_root, f"data_batch_{i}"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        imgs = batch[b"data"].astype(np.float32)/255.0
        imgs = imgs.reshape(-1, 3, 32, 32)
        x_train.append(imgs)
        y_train.extend(batch[b"labels"])
x_all = np.concatenate(x_train)
y_all = np.array(y_train)
# Use only 20000 samples
x_all = x_all[:20000]
y_all = y_all[:20000]
print(f"Data: {x_all.shape}")

# Train/val split
np.random.seed(42)
idx = np.random.permutation(len(y_all))
val_size = 5000
train_idx, val_idx = idx[val_size:], idx[:val_size]
x_tr = x_all[train_idx]; y_tr = y_all[train_idx]
x_va = x_all[val_idx]; y_va = y_all[val_idx]
print(f"Train: {x_tr.shape}, Val: {x_va.shape}")

NBATCHES = x_tr.shape[0] // BATCH

best_val_acc = 0.0; patience_counter = 0; best_epoch = 0
best_w_conv1 = None; best_w_conv2 = None; best_fc_w = None; best_fc_b = None

print(f"\nBATCH={BATCH}, LR_conv1={LR_CONV1}, LR_conv2={LR_CONV}, LR_fc={LR_FC}")
print(f"WD={WD}, Dropout={DROPOUT}, Early Stopping patience={PATIENCE}")
print(f"Data Aug: HFlip + RandomCrop")
print()

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0; correct = 0
    indices = np.random.permutation(x_tr.shape[0])
    
    for batch_idx in range(NBATCHES):
        idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
        x = augment(x_tr[indices[idx_s:idx_e]].copy())
        y = y_tr[indices[idx_s:idx_e]]
        
        # FORWARD
        d_x = lib.gpu_malloc(BATCH*C1_IN*H*W*4); lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH*C1_IN*H*W*4)
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
        
        # Dropout
        mask = (np.random.rand(BATCH, FC_IN) > DROPOUT).astype(np.float32) / (1.0 - DROPOUT)
        h_pool2_d = h_pool2 * mask
        d_fc_in_d = lib.gpu_malloc(BATCH*FC_IN*4)
        lib.gpu_memcpy_h2d(d_fc_in_d, h_pool2_d.ctypes.data, BATCH*FC_IN*4)
        
        d_fc_out = lib.gpu_malloc(BATCH*10*4)
        lib.dense_forward(d_fc_in_d, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
        h_out = np.zeros((BATCH, 10), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH*10*4)
        
        h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
        probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y)
        
        # BACKWARD
        labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
        labels_onehot[np.arange(BATCH), y] = 1.0
        d_loss = probs - labels_onehot
        
        grad_fc_w = d_loss.T @ h_pool2_d
        grad_fc_w_flat = grad_fc_w.flatten() + WD * fc_w
        grad_fc_w = grad_fc_w_flat.reshape(10, FC_IN)
        grad_pool2 = d_loss @ fc_w.reshape(10, FC_IN) * mask
        
        d_fc_grad_w = lib.gpu_malloc(10*FC_IN*4)
        lib.gpu_memcpy_h2d(d_fc_grad_w, np.clip(grad_fc_w.flatten(), -1.0, 1.0).astype(np.float32).ctypes.data, 10*FC_IN*4)
        lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR_FC), 10*FC_IN)
        h_fc_w = np.clip(g2h(d_fc_w, 10*FC_IN), -1.0, 1.0).astype(np.float32)
        lib.gpu_memcpy_h2d(d_fc_w, h_fc_w.ctypes.data, 10*FC_IN*4)
        fc_w = h_fc_w.copy()
        
        # Conv2 backward
        grad_pool2_c = np.clip(grad_pool2, -1.0, 1.0).astype(np.float32)
        d_pool2_grad = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
        lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2_c.flatten().ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)
        d_conv2_grad_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_grad_raw, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad_raw, BATCH, C2_OUT, outH2, outW2)
        d_conv2_grad = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.gpu_memset(d_conv2_grad, 0, C2_OUT*BATCH*outH2*outW2*4)
        lib.reorganize_backward(d_conv2_grad_raw, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)
        lib.leaky_relu_backward(d_conv2_raw, d_conv2_grad, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
        
        d_w_conv2_grad = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
        d_pool1_grad = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
        lib.gpu_memset(d_w_conv2_grad, 0, C2_OUT*C2_IN*KH2*KW2*4)
        lib.gpu_memset(d_pool1_grad, 0, C2_IN*BATCH*poolH1*poolW1*4)
        lib.conv_backward(d_conv2_grad, d_pool1, d_w_conv2, d_w_conv2_grad, d_pool1_grad,
                          BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
        gw2 = np.frombuffer(bytes(g2h(d_w_conv2_grad, C2_OUT*C2_IN*KH2*KW2)), dtype=np.float32).copy()
        gw2 += WD * w_conv2
        lib.gpu_memcpy_h2d(d_w_conv2_grad, np.clip(gw2, -1.0, 1.0).astype(np.float32).ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
        lib.apply_sgd_update(d_w_conv2, d_w_conv2_grad, c_float(LR_CONV), C2_OUT*C2_IN*KH2*KW2)
        h_w_conv2 = np.clip(g2h(d_w_conv2, C2_OUT*C2_IN*KH2*KW2), -2.0, 2.0).astype(np.float32)
        lib.gpu_memcpy_h2d(d_w_conv2, h_w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
        w_conv2 = h_w_conv2.copy()
        
        # Conv1 backward
        d_conv1_grad_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.gpu_memset(d_conv1_grad_raw, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.maxpool_backward_use_idx(d_pool1_grad, d_max_idx1, d_conv1_grad_raw, BATCH, C1_OUT, outH1, outW1)
        d_conv1_grad = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.gpu_memset(d_conv1_grad, 0, C1_OUT*BATCH*outH1*outW1*4)
        lib.reorganize_backward(d_conv1_grad_raw, d_conv1_grad, BATCH, C1_OUT, outH1, outW1)
        lib.leaky_relu_backward(d_conv1_raw, d_conv1_grad, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
        
        d_w_conv1_grad = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
        d_x_grad = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
        lib.gpu_memset(d_w_conv1_grad, 0, C1_OUT*C1_IN*KH1*KW1*4)
        lib.gpu_memset(d_x_grad, 0, BATCH*C1_IN*H*W*4)
        lib.conv_backward(d_conv1_grad, d_x, d_w_conv1, d_w_conv1_grad, d_x_grad,
                          BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
        gw1 = np.frombuffer(bytes(g2h(d_w_conv1_grad, C1_OUT*C1_IN*KH1*KW1)), dtype=np.float32).copy()
        gw1 += WD * w_conv1
        lib.gpu_memcpy_h2d(d_w_conv1_grad, np.clip(gw1, -1.0, 1.0).astype(np.float32).ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
        lib.apply_sgd_update(d_w_conv1, d_w_conv1_grad, c_float(LR_CONV1), C1_OUT*C1_IN*KH1*KW1)
        h_w_conv1 = np.clip(g2h(d_w_conv1, C1_OUT*C1_IN*KH1*KW1), -2.0, 2.0).astype(np.float32)
        lib.gpu_memcpy_h2d(d_w_conv1, h_w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
        w_conv1 = h_w_conv1.copy()
        
        # FREE
        lib.gpu_free(d_x); lib.gpu_free(d_col1); lib.gpu_free(d_conv1_raw); lib.gpu_free(d_conv1)
        lib.gpu_free(d_pool1); lib.gpu_free(d_max_idx1); lib.gpu_free(d_col2); lib.gpu_free(d_conv2_raw)
        lib.gpu_free(d_conv2); lib.gpu_free(d_pool2); lib.gpu_free(d_max_idx2); lib.gpu_free(d_fc_in)
        lib.gpu_free(d_fc_in_d); lib.gpu_free(d_fc_out); lib.gpu_free(d_fc_grad_w); lib.gpu_free(d_pool2_grad)
        lib.gpu_free(d_conv2_grad_raw); lib.gpu_free(d_conv2_grad); lib.gpu_free(d_w_conv2_grad)
        lib.gpu_free(d_pool1_grad); lib.gpu_free(d_conv1_grad_raw); lib.gpu_free(d_conv1_grad)
        lib.gpu_free(d_w_conv1_grad); lib.gpu_free(d_x_grad)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={loss:.4f}, acc={correct/(batch_idx+1)/BATCH*100:.1f}%")
    
    train_acc = correct / NBATCHES / BATCH * 100
    
    # Validation (limit to 200 batches)
    val_correct = 0; val_batches = min(200, x_va.shape[0] // BATCH)
    for vb in range(val_batches):
        vx = x_va[vb*BATCH:(vb+1)*BATCH]
        vy = y_va[vb*BATCH:(vb+1)*BATCH]
        vd_x = lib.gpu_malloc(BATCH*C1_IN*H*W*4); lib.gpu_memcpy_h2d(vd_x, vx.ctypes.data, BATCH*C1_IN*H*W*4)
        vd_col1 = lib.gpu_malloc(C1_IN*KH1*KW1*BATCH*outH1*outW1*4)
        vd_conv1_raw = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.im2col_forward(vd_x, vd_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
        lib.gemm_forward(d_w_conv1, vd_col1, vd_conv1_raw, C1_OUT, BATCH*outH1*outW1, C1_IN*KH1*KW1)
        lib.leaky_relu_forward(vd_conv1_raw, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
        vd_conv1 = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
        lib.reorganize_forward(vd_conv1_raw, vd_conv1, BATCH, C1_OUT, outH1, outW1)
        vd_pool1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
        vd_max_idx1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
        lib.maxpool_forward_store(vd_pool1, vd_conv1, vd_max_idx1, BATCH, C1_OUT, outH1, outW1)
        vd_col2 = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
        vd_conv2_raw = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.im2col_forward(vd_pool1, vd_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
        lib.gemm_forward(d_w_conv2, vd_col2, vd_conv2_raw, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
        lib.leaky_relu_forward(vd_conv2_raw, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
        vd_conv2 = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
        lib.reorganize_forward(vd_conv2_raw, vd_conv2, BATCH, C2_OUT, outH2, outW2)
        vd_pool2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
        vd_max_idx2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
        lib.maxpool_forward_store(vd_pool2, vd_conv2, vd_max_idx2, BATCH, C2_OUT, outH2, outW2)
        vh_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
        lib.gpu_memcpy_d2h(vh_pool2.ctypes.data, vd_pool2, BATCH*FC_IN*4)
        vd_fc_in = lib.gpu_malloc(BATCH*FC_IN*4)
        lib.gpu_memcpy_h2d(vd_fc_in, vh_pool2.ctypes.data, BATCH*FC_IN*4)
        vd_fc_out = lib.gpu_malloc(BATCH*10*4)
        lib.dense_forward(vd_fc_in, d_fc_w, d_fc_b, vd_fc_out, BATCH, FC_IN, 10)
        vh_out = np.zeros((BATCH, 10), dtype=np.float32)
        lib.gpu_memcpy_d2h(vh_out.ctypes.data, vd_fc_out, BATCH*10*4)
        vh_out_shifted = vh_out - vh_out.max(axis=1, keepdims=True)
        vprobs = np.exp(vh_out_shifted) / np.exp(vh_out_shifted).sum(axis=1, keepdims=True)
        val_correct += np.sum(np.argmax(vprobs, axis=1) == vy)
        lib.gpu_free(vd_x); lib.gpu_free(vd_col1); lib.gpu_free(vd_conv1_raw); lib.gpu_free(vd_conv1)
        lib.gpu_free(vd_pool1); lib.gpu_free(vd_max_idx1); lib.gpu_free(vd_col2); lib.gpu_free(vd_conv2_raw)
        lib.gpu_free(vd_conv2); lib.gpu_free(vd_pool2); lib.gpu_free(vd_max_idx2); lib.gpu_free(vd_fc_in)
        lib.gpu_free(vd_fc_out)
    val_acc = val_correct / val_batches / BATCH * 100
    
    print(f"Epoch {epoch+1}: Loss={total_loss/NBATCHES:.4f}, Train={train_acc:.2f}%, Val={val_acc:.2f}%, Time={time.time()-t0:.1f}s")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc; patience_counter = 0
        best_epoch = epoch + 1
        best_w_conv1 = w_conv1.copy(); best_w_conv2 = w_conv2.copy()
        best_fc_w = fc_w.copy(); best_fc_b = fc_b.copy()
        print(f"  ★ Best val_acc={val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\n=== EARLY STOPPING at epoch {epoch+1} ===")
            print(f"Best: Epoch {best_epoch}, Val={best_val_acc:.2f}%")
            break

lib.gpu_free(d_w_conv1); lib.gpu_free(d_w_conv2); lib.gpu_free(d_fc_w); lib.gpu_free(d_fc_b)
print(f"\nFinal best val_acc: {best_val_acc:.2f}%")
print("Done! V5 with Early Stopping + Weight Decay + Dropout + Data Aug (BATCH=16)")
