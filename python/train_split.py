#!/usr/bin/env python3
"""Train a small VGG-style CUDA CNN on CIFAR-10.

Architecture:
Conv(3->32) -> LeakyReLU -> Conv(32->32) -> LeakyReLU -> MaxPool
Conv(32->64) -> LeakyReLU -> Conv(64->64) -> LeakyReLU -> MaxPool
FC(1600->10)
"""
import os
import time
from ctypes import c_float

import numpy as np

from cifar10_data import load_cifar10, normalize_cifar
from cuda_backend import (
    g2h,
    lib,
    update_conv,
)
from model_forward import evaluate
from model_init import init_weights
from model_weights import free_weights, init_velocity_buffers, reload_weights_from_checkpoint, save_checkpoint, upload_weights
from train_config import (
    BATCH,
    BEST_MODEL_FILENAME,
    C1_IN,
    C1_OUT,
    C2_IN,
    C2_OUT,
    C3_IN,
    C3_OUT,
    C4_IN,
    C4_OUT,
    CONV_GRAD_SPATIAL_NORMALIZE,
    DATASET_SEED,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FC_IN,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
    GRAD_DEBUG,
    GRAD_DEBUG_BATCHES,
    GRAD_POOL_CLIP,
    H,
    H1,
    H2,
    H3,
    H4,
    INIT_SEED,
    KH,
    KW,
    LEAKY_ALPHA,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    LR_PLATEAU_PATIENCE,
    LR_REDUCE_FACTOR,
    MIN_DELTA,
    MIN_LR,
    MOMENTUM,
    N_TRAIN,
    N_VAL,
    P1H,
    P1W,
    P2H,
    P2W,
    TRAIN_BATCH_IDS,
    W,
    W1,
    W2,
    W3,
    W4,
    WEIGHT_DECAY,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_MODEL_PATH = os.path.join(ROOT, "python", BEST_MODEL_FILENAME)


def current_device_weights():
    return d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b


def current_velocity_buffers():
    return d_v_conv1, d_v_conv2, d_v_conv3, d_v_conv4, d_v_fc_w, d_v_fc_b


def malloc_floats(size):
    return lib.gpu_malloc(size * 4)


def upload_to(ptr, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)


def zero_floats(ptr, size):
    lib.gpu_memset(ptr, 0, size * 4)


def conv_forward_into(d_input_nchw, d_weight, d_col, d_raw, n, in_c, in_h, in_w, out_c):
    out_h, out_w = in_h - KH + 1, in_w - KW + 1
    lib.im2col_forward(d_input_nchw, d_col, n, in_c, in_h, in_w, KH, KW, out_h, out_w)
    lib.gemm_forward(d_weight, d_col, d_raw, out_c, n * out_h * out_w, in_c * KH * KW)
    lib.leaky_relu_forward(d_raw, c_float(LEAKY_ALPHA), out_c * n * out_h * out_w)


def maxpool_forward_into(d_input_cnhw, d_pool, d_max_idx, n, c, h, w):
    lib.maxpool_forward_store(d_pool, d_input_cnhw, d_max_idx, n, c, h, w)


def cnhw_to_nchw_into(d_cnhw, d_nchw, n, c, h, w):
    lib.cnhw_to_nchw(d_cnhw, d_nchw, n, c, h, w)


def nchw_to_cnhw_into(d_nchw, d_cnhw, n, c, h, w):
    lib.nchw_to_cnhw(d_nchw, d_cnhw, n, c, h, w)


class BatchWorkspace:
    def __init__(self):
        self.ptrs = []
        self.d_x = self.alloc(C1_IN * BATCH * H * W)
        self.d_col1 = self.alloc(C1_IN * KH * KW * BATCH * H1 * W1)
        self.d_conv1_raw = self.alloc(C1_OUT * BATCH * H1 * W1)
        self.d_conv1_nchw = self.alloc(BATCH * C1_OUT * H1 * W1)
        self.d_col2 = self.alloc(C2_IN * KH * KW * BATCH * H2 * W2)
        self.d_conv2_raw = self.alloc(C2_OUT * BATCH * H2 * W2)
        self.d_pool1 = self.alloc(C2_OUT * BATCH * P1H * P1W)
        self.d_max_idx1 = self.alloc(C2_OUT * BATCH * P1H * P1W)
        self.d_pool1_nchw = self.alloc(BATCH * C2_OUT * P1H * P1W)
        self.d_col3 = self.alloc(C3_IN * KH * KW * BATCH * H3 * W3)
        self.d_conv3_raw = self.alloc(C3_OUT * BATCH * H3 * W3)
        self.d_conv3_nchw = self.alloc(BATCH * C3_OUT * H3 * W3)
        self.d_col4 = self.alloc(C4_IN * KH * KW * BATCH * H4 * W4)
        self.d_conv4_raw = self.alloc(C4_OUT * BATCH * H4 * W4)
        self.d_pool2 = self.alloc(C4_OUT * BATCH * P2H * P2W)
        self.d_max_idx2 = self.alloc(C4_OUT * BATCH * P2H * P2W)
        self.d_pool2_nchw = self.alloc(BATCH * C4_OUT * P2H * P2W)
        self.d_fc_out = self.alloc(BATCH * 10)
        self.d_grad_logits = self.alloc(BATCH * 10)
        self.d_pool2_grad_nchw = self.alloc(BATCH * FC_IN)
        self.d_fc_grad_w = self.alloc(10 * FC_IN)
        self.d_fc_grad_b = self.alloc(10)
        self.d_pool2_grad = self.alloc(C4_OUT * BATCH * P2H * P2W)
        self.d_conv4_grad_raw = self.alloc(C4_OUT * BATCH * H4 * W4)
        self.d_w_conv4_grad = self.alloc(C4_OUT * C4_IN * KH * KW)
        self.d_conv3_grad = self.alloc(C4_IN * BATCH * H3 * W3)
        self.d_conv3_grad_raw = self.alloc(C3_OUT * BATCH * H3 * W3)
        self.d_w_conv3_grad = self.alloc(C3_OUT * C3_IN * KH * KW)
        self.d_pool1_grad = self.alloc(C3_IN * BATCH * P1H * P1W)
        self.d_pool1_grad_cnhw = self.alloc(C2_OUT * BATCH * P1H * P1W)
        self.d_conv2_grad_raw = self.alloc(C2_OUT * BATCH * H2 * W2)
        self.d_w_conv2_grad = self.alloc(C2_OUT * C2_IN * KH * KW)
        self.d_conv1_grad = self.alloc(C2_IN * BATCH * H1 * W1)
        self.d_conv1_grad_raw = self.alloc(C1_OUT * BATCH * H1 * W1)
        self.d_w_conv1_grad = self.alloc(C1_OUT * C1_IN * KH * KW)
        self.d_x_grad = self.alloc(C1_IN * BATCH * H * W)

    def alloc(self, size):
        ptr = malloc_floats(size)
        self.ptrs.append(ptr)
        return ptr

    def free(self):
        for ptr in self.ptrs:
            lib.gpu_free(ptr)
        self.ptrs = []


data_root = os.path.join(ROOT, "data", "cifar-10-batches-py")
x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_cifar10(
    data_root,
    n_train=N_TRAIN,
    n_val=N_VAL,
    seed=DATASET_SEED,
    train_batch_ids=TRAIN_BATCH_IDS,
)
x_train = normalize_cifar(x_train)
x_val = normalize_cifar(x_val)
x_test_final = normalize_cifar(x_test_final)
print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test(official): {x_test_final.shape[0]}")

w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b = init_weights(INIT_SEED)
d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = upload_weights(
    w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b
)
d_v_conv1, d_v_conv2, d_v_conv3, d_v_conv4, d_v_fc_w, d_v_fc_b = init_velocity_buffers()

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
    f"momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, BATCH={BATCH}, EPOCHS={EPOCHS}"
)
print()


NBATCHES = x_train.shape[0] // BATCH
best_val_acc = -1.0
best_epoch = -1
epochs_no_improve = 0
plateau_count = 0
current_lr_conv1 = LR_CONV1
current_lr_conv = LR_CONV
current_lr_fc = LR_FC
workspace = BatchWorkspace()

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
        upload_to(workspace.d_x, x)

        conv_forward_into(workspace.d_x, d_w_conv1, workspace.d_col1, workspace.d_conv1_raw, BATCH, C1_IN, H, W, C1_OUT)
        cnhw_to_nchw_into(workspace.d_conv1_raw, workspace.d_conv1_nchw, BATCH, C1_OUT, H1, W1)

        conv_forward_into(workspace.d_conv1_nchw, d_w_conv2, workspace.d_col2, workspace.d_conv2_raw, BATCH, C2_IN, H1, W1, C2_OUT)
        maxpool_forward_into(workspace.d_conv2_raw, workspace.d_pool1, workspace.d_max_idx1, BATCH, C2_OUT, H2, W2)
        cnhw_to_nchw_into(workspace.d_pool1, workspace.d_pool1_nchw, BATCH, C2_OUT, P1H, P1W)

        conv_forward_into(workspace.d_pool1_nchw, d_w_conv3, workspace.d_col3, workspace.d_conv3_raw, BATCH, C3_IN, P1H, P1W, C3_OUT)
        cnhw_to_nchw_into(workspace.d_conv3_raw, workspace.d_conv3_nchw, BATCH, C3_OUT, H3, W3)

        conv_forward_into(workspace.d_conv3_nchw, d_w_conv4, workspace.d_col4, workspace.d_conv4_raw, BATCH, C4_IN, H3, W3, C4_OUT)
        maxpool_forward_into(workspace.d_conv4_raw, workspace.d_pool2, workspace.d_max_idx2, BATCH, C4_OUT, H4, W4)
        cnhw_to_nchw_into(workspace.d_pool2, workspace.d_pool2_nchw, BATCH, C4_OUT, P2H, P2W)

        lib.dense_forward(workspace.d_pool2_nchw, d_fc_w, d_fc_b, workspace.d_fc_out, BATCH, FC_IN, 10)
        h_out = np.zeros((BATCH, 10), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_out.ctypes.data, workspace.d_fc_out, BATCH * 10 * 4)

        h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
        exp_out = np.exp(h_out_shifted)
        probs = exp_out / exp_out.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y)

        labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
        labels_onehot[np.arange(BATCH), y] = 1.0
        # Batch scaling belongs here. Conv updates optionally add per-layer spatial normalization.
        d_logits = (probs - labels_onehot) / BATCH

        upload_to(workspace.d_grad_logits, d_logits)
        lib.dense_backward_full(
            workspace.d_grad_logits,
            workspace.d_pool2_nchw,
            d_fc_w,
            workspace.d_pool2_grad_nchw,
            workspace.d_fc_grad_w,
            workspace.d_fc_grad_b,
            BATCH,
            FC_IN,
            10,
        )
        lib.conv_update_fused(
            d_fc_w,
            workspace.d_fc_grad_w,
            d_v_fc_w,
            c_float(current_lr_fc),
            c_float(MOMENTUM),
            c_float(WEIGHT_DECAY),
            c_float(GRAD_CLIP_FC),
            c_float(1.0),
            10 * FC_IN,
        )
        lib.conv_update_fused(
            d_fc_b,
            workspace.d_fc_grad_b,
            d_v_fc_b,
            c_float(current_lr_fc),
            c_float(MOMENTUM),
            c_float(0.0),
            c_float(GRAD_CLIP_BIAS),
            c_float(1.0),
            10,
        )

        # Conv4 backward.
        lib.clip_inplace(workspace.d_pool2_grad_nchw, c_float(GRAD_POOL_CLIP), BATCH * FC_IN)
        nchw_to_cnhw_into(workspace.d_pool2_grad_nchw, workspace.d_pool2_grad, BATCH, C4_OUT, P2H, P2W)
        zero_floats(workspace.d_conv4_grad_raw, C4_OUT * BATCH * H4 * W4)
        lib.maxpool_backward_use_idx(workspace.d_pool2_grad, workspace.d_max_idx2, workspace.d_conv4_grad_raw, BATCH, C4_OUT, H4, W4)
        lib.leaky_relu_backward(workspace.d_conv4_raw, workspace.d_conv4_grad_raw, c_float(LEAKY_ALPHA), C4_OUT * BATCH * H4 * W4)

        lib.conv_backward_precol(
            workspace.d_conv4_grad_raw, workspace.d_conv3_nchw, d_w_conv4, workspace.d_w_conv4_grad, workspace.d_conv3_grad,
            workspace.d_col4,
            BATCH, C4_IN, H3, W3, KH, KW, H4, W4, C4_OUT,
        )
        update_conv(
            d_w_conv4, workspace.d_w_conv4_grad, d_v_conv4, current_lr_conv, MOMENTUM, C4_OUT * C4_IN * KH * KW,
            "conv4", WEIGHT_DECAY, GRAD_CLIP_CONV,
            H4 * W4 if CONV_GRAD_SPATIAL_NORMALIZE else 1.0,
            log_grad,
        )

        # Conv3 backward.
        nchw_to_cnhw_into(workspace.d_conv3_grad, workspace.d_conv3_grad_raw, BATCH, C3_OUT, H3, W3)
        lib.leaky_relu_backward(workspace.d_conv3_raw, workspace.d_conv3_grad_raw, c_float(LEAKY_ALPHA), C3_OUT * BATCH * H3 * W3)
        lib.conv_backward_precol(
            workspace.d_conv3_grad_raw, workspace.d_pool1_nchw, d_w_conv3, workspace.d_w_conv3_grad, workspace.d_pool1_grad,
            workspace.d_col3,
            BATCH, C3_IN, P1H, P1W, KH, KW, H3, W3, C3_OUT,
        )
        update_conv(
            d_w_conv3, workspace.d_w_conv3_grad, d_v_conv3, current_lr_conv, MOMENTUM, C3_OUT * C3_IN * KH * KW,
            "conv3", WEIGHT_DECAY, GRAD_CLIP_CONV,
            H3 * W3 if CONV_GRAD_SPATIAL_NORMALIZE else 1.0,
            log_grad,
        )

        # Conv2 backward.
        nchw_to_cnhw_into(workspace.d_pool1_grad, workspace.d_pool1_grad_cnhw, BATCH, C2_OUT, P1H, P1W)
        zero_floats(workspace.d_conv2_grad_raw, C2_OUT * BATCH * H2 * W2)
        lib.maxpool_backward_use_idx(workspace.d_pool1_grad_cnhw, workspace.d_max_idx1, workspace.d_conv2_grad_raw, BATCH, C2_OUT, H2, W2)
        lib.leaky_relu_backward(workspace.d_conv2_raw, workspace.d_conv2_grad_raw, c_float(LEAKY_ALPHA), C2_OUT * BATCH * H2 * W2)
        lib.conv_backward_precol(
            workspace.d_conv2_grad_raw, workspace.d_conv1_nchw, d_w_conv2, workspace.d_w_conv2_grad, workspace.d_conv1_grad,
            workspace.d_col2,
            BATCH, C2_IN, H1, W1, KH, KW, H2, W2, C2_OUT,
        )
        update_conv(
            d_w_conv2, workspace.d_w_conv2_grad, d_v_conv2, current_lr_conv, MOMENTUM, C2_OUT * C2_IN * KH * KW,
            "conv2", WEIGHT_DECAY, GRAD_CLIP_CONV,
            H2 * W2 if CONV_GRAD_SPATIAL_NORMALIZE else 1.0,
            log_grad,
        )

        # Conv1 backward.
        nchw_to_cnhw_into(workspace.d_conv1_grad, workspace.d_conv1_grad_raw, BATCH, C1_OUT, H1, W1)
        lib.leaky_relu_backward(workspace.d_conv1_raw, workspace.d_conv1_grad_raw, c_float(LEAKY_ALPHA), C1_OUT * BATCH * H1 * W1)
        lib.conv_backward_precol(
            workspace.d_conv1_grad_raw, workspace.d_x, d_w_conv1, workspace.d_w_conv1_grad, workspace.d_x_grad,
            workspace.d_col1,
            BATCH, C1_IN, H, W, KH, KW, H1, W1, C1_OUT,
        )
        update_conv(
            d_w_conv1, workspace.d_w_conv1_grad, d_v_conv1, current_lr_conv1, MOMENTUM, C1_OUT * C1_IN * KH * KW,
            "conv1", WEIGHT_DECAY, GRAD_CLIP_CONV,
            H1 * W1 if CONV_GRAD_SPATIAL_NORMALIZE else 1.0,
            log_grad,
        )

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={loss:.4f}, acc={correct/(batch_idx+1)/BATCH*100:.1f}%")

    train_acc = correct / NBATCHES / BATCH * 100
    val_acc = evaluate(x_val, y_val, current_device_weights())
    epoch_loss = total_loss / NBATCHES
    elapsed = time.time() - t0
    improved = val_acc > (best_val_acc + MIN_DELTA)

    if improved:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        epochs_no_improve = 0
        plateau_count = 0
        save_checkpoint(
            BEST_MODEL_PATH,
            epoch + 1,
            val_acc,
            current_lr_conv1,
            current_lr_conv,
            current_lr_fc,
            current_device_weights(),
        )
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

workspace.free()

print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
free_weights(current_velocity_buffers())
if os.path.exists(BEST_MODEL_PATH):
    best_ckpt, fc_w, fc_b, new_device_weights = reload_weights_from_checkpoint(
        BEST_MODEL_PATH,
        current_device_weights(),
    )
    d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = new_device_weights
    print(f"Reloaded best checkpoint from epoch {int(best_ckpt['epoch'])} with Val={float(best_ckpt['val_acc']):.2f}%")
test_acc = evaluate(x_test_final, y_test_final, current_device_weights())
print(f"Test Accuracy: {test_acc:.2f}%")

free_weights(current_device_weights())
print("\nDone!")
