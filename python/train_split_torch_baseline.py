#!/usr/bin/env python3
"""PyTorch baseline with the same CIFAR-10 split, architecture, and initial weights."""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from cifar10_data import load_cifar10, normalize_cifar
from model_init import init_weights
from train_config import (
    BATCH,
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
    EVAL_MAX_BATCHES,
    FC_IN,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
    GRAD_POOL_CLIP,
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
    W1,
    W2,
    W3,
    W4,
    WEIGHT_DECAY,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_MODEL_PATH = os.path.join(ROOT, "python", "best_model_split_torch.pt")


class TorchCifarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(C1_IN, C1_OUT, kernel_size=(KH, KW), bias=False)
        self.conv2 = nn.Conv2d(C2_IN, C2_OUT, kernel_size=(KH, KW), bias=False)
        self.conv3 = nn.Conv2d(C3_IN, C3_OUT, kernel_size=(KH, KW), bias=False)
        self.conv4 = nn.Conv2d(C4_IN, C4_OUT, kernel_size=(KH, KW), bias=False)
        self.act = nn.LeakyReLU(LEAKY_ALPHA)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(FC_IN, 10)

    def forward(self, x, clamp_pool_grad=False):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)
        if clamp_pool_grad and x.requires_grad:
            x.register_hook(lambda grad: grad.clamp(-GRAD_POOL_CLIP, GRAD_POOL_CLIP))
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_initial_weights(model, device):
    w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b = init_weights(INIT_SEED)
    with torch.no_grad():
        model.conv1.weight.copy_(torch.from_numpy(w_conv1.reshape(C1_OUT, C1_IN, KH, KW)).to(device))
        model.conv2.weight.copy_(torch.from_numpy(w_conv2.reshape(C2_OUT, C2_IN, KH, KW)).to(device))
        model.conv3.weight.copy_(torch.from_numpy(w_conv3.reshape(C3_OUT, C3_IN, KH, KW)).to(device))
        model.conv4.weight.copy_(torch.from_numpy(w_conv4.reshape(C4_OUT, C4_IN, KH, KW)).to(device))
        model.fc.weight.copy_(torch.from_numpy(fc_w.reshape(10, FC_IN)).to(device))
        model.fc.bias.copy_(torch.from_numpy(fc_b).to(device))


def apply_sgd_update(model, lr_conv1, lr_conv, lr_fc):
    updates = [
        (model.conv1.weight, lr_conv1, GRAD_CLIP_CONV, True),
        (model.conv2.weight, lr_conv, GRAD_CLIP_CONV, True),
        (model.conv3.weight, lr_conv, GRAD_CLIP_CONV, True),
        (model.conv4.weight, lr_conv, GRAD_CLIP_CONV, True),
        (model.fc.weight, lr_fc, GRAD_CLIP_FC, True),
        (model.fc.bias, lr_fc, GRAD_CLIP_BIAS, False),
    ]
    with torch.no_grad():
        for param, lr, clip_value, use_decay in updates:
            grad = param.grad
            if use_decay:
                grad = grad + WEIGHT_DECAY * param
            grad = torch.clamp(grad, -clip_value, clip_value)
            param -= lr * grad
            param.grad = None


def evaluate(model, x, y, device, batch_size=BATCH, max_batches=EVAL_MAX_BATCHES):
    model.eval()
    correct = 0
    total = 0
    nbatches = min(x.shape[0] // batch_size, max_batches)
    with torch.no_grad():
        for i in range(nbatches):
            idx_s = i * batch_size
            idx_e = idx_s + batch_size
            xb = torch.from_numpy(x[idx_s:idx_e]).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            correct += np.sum(pred == y[idx_s:idx_e])
            total += batch_size
    model.train()
    return correct / total * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INIT_SEED)

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

    model = TorchCifarCnn().to(device)
    load_initial_weights(model, device)
    criterion = nn.CrossEntropyLoss()

    nbatches = x_train.shape[0] // BATCH
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    plateau_count = 0
    current_lr_conv1 = LR_CONV1
    current_lr_conv = LR_CONV
    current_lr_fc = LR_FC

    print(f"Device: {device}")
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test(official): {x_test_final.shape[0]}")
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
    print(f"DATASET_SEED={DATASET_SEED}, INIT_SEED={INIT_SEED}")
    print()

    for epoch in range(EPOCHS):
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        indices = np.random.permutation(x_train.shape[0])

        for batch_idx in range(nbatches):
            idx_s = batch_idx * BATCH
            idx_e = idx_s + BATCH
            x = x_train[indices[idx_s:idx_e]]
            y = y_train[indices[idx_s:idx_e]]

            if np.random.rand() > 0.5:
                x = x[:, :, :, ::-1].copy()

            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(y.astype(np.int64, copy=False)).to(device)

            logits = model(xb, clamp_pool_grad=True)
            loss = criterion(logits, yb)
            loss.backward()
            apply_sgd_update(model, current_lr_conv1, current_lr_conv, current_lr_fc)

            total_loss += float(loss.detach().cpu().item())
            pred = torch.argmax(logits.detach(), dim=1)
            correct += int((pred == yb).sum().detach().cpu().item())

            if (batch_idx + 1) % 100 == 0:
                acc = correct / (batch_idx + 1) / BATCH * 100
                print(f"  Batch {batch_idx+1}/{nbatches}: loss={float(loss.detach().cpu().item()):.4f}, acc={acc:.1f}%")

        train_acc = correct / nbatches / BATCH * 100
        val_acc = evaluate(model, x_val, y_val, device)
        epoch_loss = total_loss / nbatches
        elapsed = time.time() - t0
        improved = val_acc > (best_val_acc + MIN_DELTA)

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            plateau_count = 0
            torch.save(
                {
                    "epoch": int(epoch + 1),
                    "val_acc": float(val_acc),
                    "lr_conv1": float(current_lr_conv1),
                    "lr_conv": float(current_lr_conv),
                    "lr_fc": float(current_lr_fc),
                    "model_state": model.state_dict(),
                },
                BEST_MODEL_PATH,
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
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Reloaded best checkpoint from epoch {int(ckpt['epoch'])} with Val={float(ckpt['val_acc']):.2f}%")
    test_acc = evaluate(model, x_test_final, y_test_final, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
