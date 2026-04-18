#!/usr/bin/env python3
"""Train a small VGG-style CUDA CNN on CIFAR-10.

Architecture:
Conv(3->32) -> LeakyReLU -> Conv(32->32) -> LeakyReLU -> MaxPool
Conv(32->64) -> LeakyReLU -> Conv(64->64) -> LeakyReLU -> MaxPool
FC(1600->10)
"""
import ctypes
import os
import pickle
import time
from ctypes import c_float, c_int, c_void_p

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
so = os.path.join(ROOT, "cpp", "libminimal_cuda_cnn.so")
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
lib.nchw_to_cnhw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.cnhw_to_nchw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
]

BATCH = 64
EPOCHS = 50
LR_CONV1 = 0.003
LR_CONV = 0.003
LR_FC = 0.003
LEAKY_ALPHA = 0.1
GRAD_DEBUG = os.environ.get("GRAD_DEBUG") == "1"
GRAD_DEBUG_BATCHES = int(os.environ.get("GRAD_DEBUG_BATCHES", "1"))
WEIGHT_DECAY = 5e-4
GRAD_CLIP_CONV = 1.0
GRAD_CLIP_FC = 1.0
GRAD_CLIP_BIAS = 5.0
LR_PLATEAU_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-5
EARLY_STOP_PATIENCE = 8
MIN_DELTA = 0.05
BEST_MODEL_PATH = os.path.join(ROOT, "python", "best_model_split.npz")

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

C1_IN, C1_OUT = 3, 32
C2_IN, C2_OUT = 32, 32
C3_IN, C3_OUT = 32, 64
C4_IN, C4_OUT = 64, 64
H, W = 32, 32
KH, KW = 3, 3

H1, W1 = H - KH + 1, W - KW + 1          # 30x30
H2, W2 = H1 - KH + 1, W1 - KW + 1        # 28x28
P1H, P1W = H2 // 2, W2 // 2              # 14x14
H3, W3 = P1H - KH + 1, P1W - KW + 1      # 12x12
H4, W4 = H3 - KH + 1, W3 - KW + 1        # 10x10
P2H, P2W = H4 // 2, W4 // 2              # 5x5
FC_IN = C4_OUT * P2H * P2W


def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h


def normalize_cifar(x):
    return ((x - CIFAR_MEAN) / CIFAR_STD).astype(np.float32)


def gpu_zeros(size):
    ptr = lib.gpu_malloc(size * 4)
    lib.gpu_memset(ptr, 0, size * 4)
    return ptr


def upload(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    ptr = lib.gpu_malloc(arr.size * 4)
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)
    return ptr


def cnhw_to_nchw_alloc(d_cnhw, n, c, h, w):
    d_nchw = lib.gpu_malloc(n * c * h * w * 4)
    lib.cnhw_to_nchw(d_cnhw, d_nchw, n, c, h, w)
    return d_nchw


def nchw_to_cnhw_alloc(d_nchw, n, c, h, w):
    d_cnhw = lib.gpu_malloc(n * c * h * w * 4)
    lib.nchw_to_cnhw(d_nchw, d_cnhw, n, c, h, w)
    return d_cnhw


def conv_forward(d_input_nchw, d_weight, n, in_c, in_h, in_w, out_c):
    out_h, out_w = in_h - KH + 1, in_w - KW + 1
    col_size = in_c * KH * KW * n * out_h * out_w
    raw_size = out_c * n * out_h * out_w
    d_col = lib.gpu_malloc(col_size * 4)
    d_raw = lib.gpu_malloc(raw_size * 4)
    lib.im2col_forward(d_input_nchw, d_col, n, in_c, in_h, in_w, KH, KW, out_h, out_w)
    lib.gemm_forward(d_weight, d_col, d_raw, out_c, n * out_h * out_w, in_c * KH * KW)
    lib.leaky_relu_forward(d_raw, c_float(LEAKY_ALPHA), raw_size)
    return d_col, d_raw, out_h, out_w


def maxpool_forward(d_input_cnhw, n, c, h, w):
    out_h, out_w = h // 2, w // 2
    out_size = c * n * out_h * out_w
    d_pool = lib.gpu_malloc(out_size * 4)
    d_idx = lib.gpu_malloc(out_size * 4)
    lib.maxpool_forward_store(d_pool, d_input_cnhw, d_idx, n, c, h, w)
    return d_pool, d_idx, out_h, out_w


def update_conv(d_weight, d_grad, lr, size, name, weight_decay, clip_value, log_grad=False):
    h_grad = g2h(d_grad, size).reshape(-1)
    h_weight = g2h(d_weight, size).reshape(-1)
    h_grad = h_grad + weight_decay * h_weight
    if log_grad:
        print(f"    {name} grad_abs_mean={np.mean(np.abs(h_grad)):.6e} grad_abs_max={np.max(np.abs(h_grad)):.6e}")
    h_grad_clip = np.clip(h_grad, -clip_value, clip_value).astype(np.float32)
    lib.gpu_memcpy_h2d(d_grad, h_grad_clip.ctypes.data, size * 4)
    lib.apply_sgd_update(d_weight, d_grad, c_float(lr), size)


def save_checkpoint(path, epoch, val_acc, lr_conv1, lr_conv, lr_fc):
    np.savez(
        path,
        epoch=np.int32(epoch),
        val_acc=np.float32(val_acc),
        lr_conv1=np.float32(lr_conv1),
        lr_conv=np.float32(lr_conv),
        lr_fc=np.float32(lr_fc),
        w_conv1=g2h(d_w_conv1, C1_OUT * C1_IN * KH * KW),
        w_conv2=g2h(d_w_conv2, C2_OUT * C2_IN * KH * KW),
        w_conv3=g2h(d_w_conv3, C3_OUT * C3_IN * KH * KW),
        w_conv4=g2h(d_w_conv4, C4_OUT * C4_IN * KH * KW),
        fc_w=g2h(d_fc_w, 10 * FC_IN),
        fc_b=g2h(d_fc_b, 10),
    )


def reload_weights_from_checkpoint(path):
    global d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b
    global fc_w, fc_b

    ckpt = np.load(path)
    for ptr in [d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b]:
        lib.gpu_free(ptr)

    fc_w = ckpt["fc_w"].astype(np.float32)
    fc_b = ckpt["fc_b"].astype(np.float32)
    d_w_conv1 = upload(ckpt["w_conv1"].astype(np.float32))
    d_w_conv2 = upload(ckpt["w_conv2"].astype(np.float32))
    d_w_conv3 = upload(ckpt["w_conv3"].astype(np.float32))
    d_w_conv4 = upload(ckpt["w_conv4"].astype(np.float32))
    d_fc_w = upload(fc_w)
    d_fc_b = upload(fc_b)
    return ckpt


def load_cifar10():
    data_root = os.path.join(ROOT, "data", "cifar-10-batches-py")
    print("Loading 5 training batches...")
    x_parts = []
    y_parts = []
    for i in range(1, 6):
        with open(os.path.join(data_root, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x_parts.append(batch[b"data"].astype(np.float32) / 255.0)
        y_parts.append(np.array(batch[b"labels"]))
    x_train_all = np.concatenate(x_parts, axis=0).reshape(-1, 3, 32, 32)
    y_train_all = np.concatenate(y_parts, axis=0)
    print(f"Total training samples: {x_train_all.shape[0]}")

    with open(os.path.join(data_root, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x_test = (batch[b"data"].astype(np.float32) / 255.0).reshape(-1, 3, 32, 32)
    y_test = np.array(batch[b"labels"])
    print(f"Test samples: {x_test.shape[0]}")

    print("Loading data_batch_1 only...")
    with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x_train_all = (batch[b"data"].astype(np.float32) / 255.0).reshape(-1, 3, 32, 32)
    y_train_all = np.array(batch[b"labels"])
    print(f"Training samples: {x_train_all.shape[0]}")

    n_train = 8000
    n_val = 2000
    indices = np.random.permutation(x_train_all.shape[0])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    return (
        x_train_all[train_idx],
        y_train_all[train_idx],
        x_train_all[val_idx],
        y_train_all[val_idx],
        x_test,
        y_test,
    )


x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_cifar10()
x_train = normalize_cifar(x_train)
x_val = normalize_cifar(x_val)
x_test_final = normalize_cifar(x_test_final)
print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test(official): {x_test_final.shape[0]}")

np.random.seed(42)

def he_init(size, fan_in):
    return np.random.randn(size).astype(np.float32) * np.sqrt(2.0 / fan_in)


w_conv1 = he_init(C1_OUT * C1_IN * KH * KW, C1_IN * KH * KW)
w_conv2 = he_init(C2_OUT * C2_IN * KH * KW, C2_IN * KH * KW)
w_conv3 = he_init(C3_OUT * C3_IN * KH * KW, C3_IN * KH * KW)
w_conv4 = he_init(C4_OUT * C4_IN * KH * KW, C4_IN * KH * KW)
fc_w = he_init(10 * FC_IN, FC_IN)
fc_b = np.zeros(10, dtype=np.float32)

d_w_conv1 = upload(w_conv1)
d_w_conv2 = upload(w_conv2)
d_w_conv3 = upload(w_conv3)
d_w_conv4 = upload(w_conv4)
d_fc_w = upload(fc_w)
d_fc_b = upload(fc_b)

print(
    "Arch: Conv1(3->32)->Conv2(32->32)->Pool1"
    f"->Conv3(32->64)->Conv4(64->64)->Pool2->FC({FC_IN}->10)"
)
print(
    f"Shapes: 32x32 -> {H1}x{W1} -> {H2}x{W2} -> {P1H}x{P1W}"
    f" -> {H3}x{W3} -> {H4}x{W4} -> {P2H}x{P2W}"
)
print(
    f"LR_conv1={LR_CONV1}, LR_conv={LR_CONV}, LR_fc={LR_FC}, "
    f"weight_decay={WEIGHT_DECAY}, BATCH={BATCH}, EPOCHS={EPOCHS}"
)
print()


def forward_batch(x):
    n = x.shape[0]
    d_x = upload(x)

    d_col1, d_conv1_raw, _, _ = conv_forward(d_x, d_w_conv1, n, C1_IN, H, W, C1_OUT)
    d_conv1_nchw = cnhw_to_nchw_alloc(d_conv1_raw, n, C1_OUT, H1, W1)

    d_col2, d_conv2_raw, _, _ = conv_forward(d_conv1_nchw, d_w_conv2, n, C2_IN, H1, W1, C2_OUT)
    d_pool1, d_max_idx1, _, _ = maxpool_forward(d_conv2_raw, n, C2_OUT, H2, W2)
    d_pool1_nchw = cnhw_to_nchw_alloc(d_pool1, n, C2_OUT, P1H, P1W)

    d_col3, d_conv3_raw, _, _ = conv_forward(d_pool1_nchw, d_w_conv3, n, C3_IN, P1H, P1W, C3_OUT)
    d_conv3_nchw = cnhw_to_nchw_alloc(d_conv3_raw, n, C3_OUT, H3, W3)

    d_col4, d_conv4_raw, _, _ = conv_forward(d_conv3_nchw, d_w_conv4, n, C4_IN, H3, W3, C4_OUT)
    d_pool2, d_max_idx2, _, _ = maxpool_forward(d_conv4_raw, n, C4_OUT, H4, W4)
    d_pool2_nchw = cnhw_to_nchw_alloc(d_pool2, n, C4_OUT, P2H, P2W)

    h_pool2 = np.zeros((n, FC_IN), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2_nchw, n * FC_IN * 4)

    d_fc_in = upload(h_pool2)
    d_fc_out = lib.gpu_malloc(n * 10 * 4)
    lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, n, FC_IN, 10)
    h_out = np.zeros((n, 10), dtype=np.float32)
    lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, n * 10 * 4)

    for ptr in [
        d_x, d_col1, d_conv1_raw, d_conv1_nchw,
        d_col2, d_conv2_raw, d_pool1, d_max_idx1, d_pool1_nchw,
        d_col3, d_conv3_raw, d_conv3_nchw,
        d_col4, d_conv4_raw, d_pool2, d_max_idx2, d_pool2_nchw,
        d_fc_in, d_fc_out,
    ]:
        lib.gpu_free(ptr)

    return h_out


def evaluate(x, y, batch_size=BATCH, max_batches=50):
    correct = 0
    total = 0
    nbatches = min(x.shape[0] // batch_size, max_batches)
    for i in range(nbatches):
        idx_s = i * batch_size
        idx_e = idx_s + batch_size
        h_out = forward_batch(x[idx_s:idx_e])
        correct += np.sum(np.argmax(h_out, axis=1) == y[idx_s:idx_e])
        total += batch_size
    return correct / total * 100


NBATCHES = x_train.shape[0] // BATCH
best_val_acc = -1.0
best_epoch = -1
epochs_no_improve = 0
plateau_count = 0
current_lr_conv1 = LR_CONV1
current_lr_conv = LR_CONV
current_lr_fc = LR_FC

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0
    correct = 0
    indices = np.random.permutation(x_train.shape[0])

    for batch_idx in range(NBATCHES):
        log_grad = GRAD_DEBUG and batch_idx < GRAD_DEBUG_BATCHES
        idx_s = batch_idx * BATCH
        idx_e = idx_s + BATCH
        x = x_train[indices[idx_s:idx_e]]
        y = y_train[indices[idx_s:idx_e]]

        if np.random.rand() > 0.5:
            x = x[:, :, :, ::-1].copy()

        # Forward, keeping intermediates needed for backward.
        d_x = upload(x)

        d_col1, d_conv1_raw, _, _ = conv_forward(d_x, d_w_conv1, BATCH, C1_IN, H, W, C1_OUT)
        d_conv1_nchw = cnhw_to_nchw_alloc(d_conv1_raw, BATCH, C1_OUT, H1, W1)

        d_col2, d_conv2_raw, _, _ = conv_forward(d_conv1_nchw, d_w_conv2, BATCH, C2_IN, H1, W1, C2_OUT)
        d_pool1, d_max_idx1, _, _ = maxpool_forward(d_conv2_raw, BATCH, C2_OUT, H2, W2)
        d_pool1_nchw = cnhw_to_nchw_alloc(d_pool1, BATCH, C2_OUT, P1H, P1W)

        d_col3, d_conv3_raw, _, _ = conv_forward(d_pool1_nchw, d_w_conv3, BATCH, C3_IN, P1H, P1W, C3_OUT)
        d_conv3_nchw = cnhw_to_nchw_alloc(d_conv3_raw, BATCH, C3_OUT, H3, W3)

        d_col4, d_conv4_raw, _, _ = conv_forward(d_conv3_nchw, d_w_conv4, BATCH, C4_IN, H3, W3, C4_OUT)
        d_pool2, d_max_idx2, _, _ = maxpool_forward(d_conv4_raw, BATCH, C4_OUT, H4, W4)
        d_pool2_nchw = cnhw_to_nchw_alloc(d_pool2, BATCH, C4_OUT, P2H, P2W)

        h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2_nchw, BATCH * FC_IN * 4)
        d_fc_in = upload(h_pool2)
        d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
        lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
        h_out = np.zeros((BATCH, 10), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)

        h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
        exp_out = np.exp(h_out_shifted)
        probs = exp_out / exp_out.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y)

        labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
        labels_onehot[np.arange(BATCH), y] = 1.0
        # Mean cross-entropy scaling belongs here. Do not divide downstream conv gradients again.
        d_logits = (probs - labels_onehot) / BATCH

        grad_fc_w = d_logits.T @ h_pool2
        grad_fc_b = d_logits.sum(axis=0)
        grad_pool2 = d_logits @ fc_w.reshape(10, FC_IN)

        grad_fc_w = grad_fc_w + WEIGHT_DECAY * fc_w.reshape(10, FC_IN)
        grad_fc_w = np.clip(grad_fc_w, -GRAD_CLIP_FC, GRAD_CLIP_FC).astype(np.float32)
        grad_fc_b = np.clip(grad_fc_b, -GRAD_CLIP_BIAS, GRAD_CLIP_BIAS).astype(np.float32)

        d_fc_grad_w = upload(grad_fc_w)
        lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(current_lr_fc), 10 * FC_IN)
        d_fc_grad_b = upload(grad_fc_b)
        lib.apply_sgd_update(d_fc_b, d_fc_grad_b, c_float(current_lr_fc), 10)
        lib.gpu_free(d_fc_grad_w)
        lib.gpu_free(d_fc_grad_b)
        fc_w = g2h(d_fc_w, 10 * FC_IN).astype(np.float32)
        fc_b = g2h(d_fc_b, 10).astype(np.float32)

        # Conv4 backward.
        grad_pool2_cnhw = np.transpose(
            np.clip(grad_pool2, -1.0, 1.0).astype(np.float32).reshape(BATCH, C4_OUT, P2H, P2W),
            (1, 0, 2, 3),
        ).copy()
        d_pool2_grad = upload(grad_pool2_cnhw)
        d_conv4_grad_raw = gpu_zeros(C4_OUT * BATCH * H4 * W4)
        lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv4_grad_raw, BATCH, C4_OUT, H4, W4)
        lib.leaky_relu_backward(d_conv4_raw, d_conv4_grad_raw, c_float(LEAKY_ALPHA), C4_OUT * BATCH * H4 * W4)

        d_w_conv4_grad = gpu_zeros(C4_OUT * C4_IN * KH * KW)
        d_conv3_grad = gpu_zeros(C4_IN * BATCH * H3 * W3)
        lib.conv_backward(
            d_conv4_grad_raw, d_conv3_nchw, d_w_conv4, d_w_conv4_grad, d_conv3_grad,
            BATCH, C4_IN, H3, W3, KH, KW, H4, W4, C4_OUT,
        )
        update_conv(
            d_w_conv4, d_w_conv4_grad, current_lr_conv, C4_OUT * C4_IN * KH * KW,
            "conv4", WEIGHT_DECAY, GRAD_CLIP_CONV, log_grad,
        )

        # Conv3 backward.
        d_conv3_grad_raw = nchw_to_cnhw_alloc(d_conv3_grad, BATCH, C3_OUT, H3, W3)
        lib.leaky_relu_backward(d_conv3_raw, d_conv3_grad_raw, c_float(LEAKY_ALPHA), C3_OUT * BATCH * H3 * W3)
        d_w_conv3_grad = gpu_zeros(C3_OUT * C3_IN * KH * KW)
        d_pool1_grad = gpu_zeros(C3_IN * BATCH * P1H * P1W)
        lib.conv_backward(
            d_conv3_grad_raw, d_pool1_nchw, d_w_conv3, d_w_conv3_grad, d_pool1_grad,
            BATCH, C3_IN, P1H, P1W, KH, KW, H3, W3, C3_OUT,
        )
        update_conv(
            d_w_conv3, d_w_conv3_grad, current_lr_conv, C3_OUT * C3_IN * KH * KW,
            "conv3", WEIGHT_DECAY, GRAD_CLIP_CONV, log_grad,
        )

        # Conv2 backward.
        d_pool1_grad_cnhw = nchw_to_cnhw_alloc(d_pool1_grad, BATCH, C2_OUT, P1H, P1W)
        d_conv2_grad_raw = gpu_zeros(C2_OUT * BATCH * H2 * W2)
        lib.maxpool_backward_use_idx(d_pool1_grad_cnhw, d_max_idx1, d_conv2_grad_raw, BATCH, C2_OUT, H2, W2)
        lib.leaky_relu_backward(d_conv2_raw, d_conv2_grad_raw, c_float(LEAKY_ALPHA), C2_OUT * BATCH * H2 * W2)
        d_w_conv2_grad = gpu_zeros(C2_OUT * C2_IN * KH * KW)
        d_conv1_grad = gpu_zeros(C2_IN * BATCH * H1 * W1)
        lib.conv_backward(
            d_conv2_grad_raw, d_conv1_nchw, d_w_conv2, d_w_conv2_grad, d_conv1_grad,
            BATCH, C2_IN, H1, W1, KH, KW, H2, W2, C2_OUT,
        )
        update_conv(
            d_w_conv2, d_w_conv2_grad, current_lr_conv, C2_OUT * C2_IN * KH * KW,
            "conv2", WEIGHT_DECAY, GRAD_CLIP_CONV, log_grad,
        )

        # Conv1 backward.
        d_conv1_grad_raw = nchw_to_cnhw_alloc(d_conv1_grad, BATCH, C1_OUT, H1, W1)
        lib.leaky_relu_backward(d_conv1_raw, d_conv1_grad_raw, c_float(LEAKY_ALPHA), C1_OUT * BATCH * H1 * W1)
        d_w_conv1_grad = gpu_zeros(C1_OUT * C1_IN * KH * KW)
        d_x_grad = gpu_zeros(C1_IN * BATCH * H * W)
        lib.conv_backward(
            d_conv1_grad_raw, d_x, d_w_conv1, d_w_conv1_grad, d_x_grad,
            BATCH, C1_IN, H, W, KH, KW, H1, W1, C1_OUT,
        )
        update_conv(
            d_w_conv1, d_w_conv1_grad, current_lr_conv1, C1_OUT * C1_IN * KH * KW,
            "conv1", WEIGHT_DECAY, GRAD_CLIP_CONV, log_grad,
        )

        for ptr in [
            d_x, d_col1, d_conv1_raw, d_conv1_nchw,
            d_col2, d_conv2_raw, d_pool1, d_max_idx1, d_pool1_nchw,
            d_col3, d_conv3_raw, d_conv3_nchw,
            d_col4, d_conv4_raw, d_pool2, d_max_idx2, d_pool2_nchw,
            d_fc_in, d_fc_out, d_pool2_grad, d_conv4_grad_raw,
            d_w_conv4_grad, d_conv3_grad, d_conv3_grad_raw,
            d_w_conv3_grad, d_pool1_grad, d_pool1_grad_cnhw,
            d_conv2_grad_raw, d_w_conv2_grad, d_conv1_grad,
            d_conv1_grad_raw, d_w_conv1_grad, d_x_grad,
        ]:
            lib.gpu_free(ptr)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={loss:.4f}, acc={correct/(batch_idx+1)/BATCH*100:.1f}%")

    train_acc = correct / NBATCHES / BATCH * 100
    val_acc = evaluate(x_val, y_val)
    epoch_loss = total_loss / NBATCHES
    elapsed = time.time() - t0
    improved = val_acc > (best_val_acc + MIN_DELTA)

    if improved:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        epochs_no_improve = 0
        plateau_count = 0
        save_checkpoint(BEST_MODEL_PATH, epoch + 1, val_acc, current_lr_conv1, current_lr_conv, current_lr_fc)
        save_msg = " [saved best]"
    else:
        epochs_no_improve += 1
        plateau_count += 1
        save_msg = ""

    if plateau_count >= LR_PLATEAU_PATIENCE:
        new_lr_conv1 = max(current_lr_conv1 * LR_REDUCE_FACTOR, MIN_LR)
        new_lr_conv = max(current_lr_conv * LR_REDUCE_FACTOR, MIN_LR)
        new_lr_fc = max(current_lr_fc * LR_REDUCE_FACTOR, MIN_LR)
        if (new_lr_conv1, new_lr_conv, new_lr_fc) != (current_lr_conv1, current_lr_conv, current_lr_fc):
            current_lr_conv1 = new_lr_conv1
            current_lr_conv = new_lr_conv
            current_lr_fc = new_lr_fc
            print(
                f"  LR reduced -> conv1={current_lr_conv1:.6f}, "
                f"conv={current_lr_conv:.6f}, fc={current_lr_fc:.6f}"
            )
        plateau_count = 0

    print(
        f"Epoch {epoch+1}/{EPOCHS}: Loss={epoch_loss:.4f}, Train={train_acc:.2f}%, Val={val_acc:.2f}%, "
        f"BestVal={best_val_acc:.2f}% @ {best_epoch}, "
        f"LRs=({current_lr_conv1:.6f}, {current_lr_conv:.6f}, {current_lr_fc:.6f}), "
        f"Time={elapsed:.1f}s{save_msg}"
    )

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping after {epoch+1} epochs; best val {best_val_acc:.2f}% at epoch {best_epoch}.")
        break

print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
if os.path.exists(BEST_MODEL_PATH):
    best_ckpt = reload_weights_from_checkpoint(BEST_MODEL_PATH)
    print(f"Reloaded best checkpoint from epoch {int(best_ckpt['epoch'])} with Val={float(best_ckpt['val_acc']):.2f}%")
test_acc = evaluate(x_test_final, y_test_final)
print(f"Test Accuracy: {test_acc:.2f}%")

for ptr in [d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b]:
    lib.gpu_free(ptr)
print("\nDone!")
