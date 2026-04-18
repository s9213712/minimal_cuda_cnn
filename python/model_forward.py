"""Forward pass and evaluation helpers for the CIFAR-10 CUDA CNN."""

import numpy as np

from cuda_backend import cnhw_to_nchw_alloc, conv_forward, lib, maxpool_forward, upload
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
    EVAL_MAX_BATCHES,
    FC_IN,
    H,
    H1,
    H2,
    H3,
    H4,
    P1H,
    P1W,
    P2H,
    P2W,
    W,
    W1,
    W2,
    W3,
    W4,
)


def forward_batch(x, device_weights):
    d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = device_weights
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


def evaluate(x, y, device_weights, batch_size=BATCH, max_batches=EVAL_MAX_BATCHES):
    correct = 0
    total = 0
    nbatches = min(x.shape[0] // batch_size, max_batches)
    for i in range(nbatches):
        idx_s = i * batch_size
        idx_e = idx_s + batch_size
        h_out = forward_batch(x[idx_s:idx_e], device_weights)
        correct += np.sum(np.argmax(h_out, axis=1) == y[idx_s:idx_e])
        total += batch_size
    return correct / total * 100
