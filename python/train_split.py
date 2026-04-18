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
    cnhw_to_nchw_alloc,
    conv_forward,
    g2h,
    gpu_zeros,
    lib,
    maxpool_forward,
    nchw_to_cnhw_alloc,
    update_conv,
    upload,
)
from model_forward import evaluate
from model_init import init_weights
from model_weights import free_weights, reload_weights_from_checkpoint, save_checkpoint, upload_weights
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
            np.clip(grad_pool2, -GRAD_POOL_CLIP, GRAD_POOL_CLIP).astype(np.float32).reshape(BATCH, C4_OUT, P2H, P2W),
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

print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
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
