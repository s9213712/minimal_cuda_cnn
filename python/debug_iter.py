#!/usr/bin/env python3
"""Debug single training iteration"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int, c_float
import pickle

class Bridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = c_void_p
        self.lib.gpu_free.argtypes = [c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        self.lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    print("=== Single Iteration Debug ===")
    
    bridge = Bridge(so_path)
    
    # Load a batch
    with open(f"{data_root}/data_batch_1", 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x = batch[b'data'][:8].astype(np.float32) / 255.0
        y = np.array(batch[b'labels'][:8])
    
    x = x.reshape(-1, 3, 32, 32)
    print(f"Input: {x.shape}")
    
    BATCH, C, H, W = 8, 3, 32, 32
    KH, KW, OC = 3, 3, 32
    FC_IN = OC * 15 * 15
    LR = 0.01
    
    # Init weights
    w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * np.sqrt(2.0 / (C * KH * KW + OC * KH * KW))
    fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / (FC_IN + 10))
    fc_b = np.zeros(10, dtype=np.float32)
    
    d_w_conv = bridge.malloc(OC * C * KH * KW * 4)
    d_fc_w = bridge.malloc(10 * FC_IN * 4)
    d_fc_b = bridge.malloc(10 * 4)
    bridge.h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
    bridge.h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
    bridge.h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    
    # Forward Conv
    print("Conv forward...")
    outH, outW = 30, 30
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    d_conv_out = bridge.malloc(OC * BATCH * outH * outW * 4)
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    bridge.lib.gemm_forward(d_w_conv, d_col, d_conv_out, OC, BATCH * outH * outW, C * KH * KW)
    bridge.lib.apply_relu(d_conv_out, OC * BATCH * outH * outW)
    
    # Pool
    print("Pool forward...")
    d_pool_out = bridge.malloc(OC * BATCH * 15 * 15 * 4)
    bridge.lib.apply_maxpool(d_conv_out, d_pool_out, BATCH, OC, outH, outW)
    
    # Flatten
    h_pool = np.zeros((BATCH, FC_IN), dtype=np.float32)
    bridge.d2h(h_pool.ctypes.data, d_pool_out, BATCH * FC_IN * 4)
    print(f"Pool output: min={h_pool.min():.4f}, max={h_pool.max():.4f}, anyNaN={np.isnan(h_pool).any()}")
    
    d_fc_in = bridge.malloc(BATCH * FC_IN * 4)
    bridge.h2d(d_fc_in, h_pool.ctypes.data, BATCH * FC_IN * 4)
    
    # FC
    print("FC forward...")
    d_fc_out = bridge.malloc(BATCH * 10 * 4)
    bridge.lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    
    # Loss
    print("Computing loss...")
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
    print(f"FC output: min={h_out.min():.4f}, max={h_out.max():.4f}, anyNaN={np.isnan(h_out).any()}")
    
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    exp_out = np.exp(h_out_shifted)
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)
    print(f"probs: min={probs.min():.4f}, max={probs.max():.4f}, anyNaN={np.isnan(probs).any()}")
    
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    print(f"Loss: {loss:.4f}")
    
    # Backward
    print("Backward...")
    labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
    labels_onehot[np.arange(BATCH), y] = 1.0
    d_loss = probs - labels_onehot
    
    # FC gradient
    grad_fc_w = d_loss.T @ h_pool
    grad_fc_w = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
    
    d_fc_grad_w = bridge.malloc(10 * FC_IN * 4)
    bridge.h2d(d_fc_grad_w, grad_fc_w.ctypes.data, 10 * FC_IN * 4)
    bridge.lib.apply_sgd_update(d_fc_w, d_fc_grad_w, LR, 10 * FC_IN)
    
    print("Done!")
    
    # Cleanup
    bridge.free(d_x)
    bridge.free(d_col)
    bridge.free(d_conv_out)
    bridge.free(d_pool_out)
    bridge.free(d_fc_in)
    bridge.free(d_fc_out)
    bridge.free(d_fc_grad_w)
    bridge.free(d_w_conv)
    bridge.free(d_fc_w)
    bridge.free(d_fc_b)


if __name__ == '__main__':
    test()
