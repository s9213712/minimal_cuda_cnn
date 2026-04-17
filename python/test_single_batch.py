#!/usr/bin/env python3
"""Test single batch with full backward"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
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
        self.lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        self.lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        self.lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.apply_relu_backward.argtypes = [c_void_p, c_void_p, c_int]
        self.lib.maxpool_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        self.lib.maxpool_backward_nchw.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    print("=== Single Batch Test ===")
    
    bridge = Bridge(so_path)
    
    with open(f"{data_root}/data_batch_1", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        x = batch[b"data"][:64].astype(np.float32) / 255.0
        y = np.array(batch[b"labels"][:64])
    x = x.reshape(-1, 3, 32, 32)
    print(f"Input: {x.shape}")
    
    BATCH, C, H, W = 64, 3, 32, 32
    KH, KW, OC = 3, 3, 32
    outH, outW = H - KH + 1, W - KW + 1
    FC_IN = OC * 15 * 15
    
    np.random.seed(42)
    w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.001
    fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * 0.001
    fc_b = np.zeros(10, dtype=np.float32)
    
    d_w_conv = bridge.malloc(OC * C * KH * KW * 4)
    d_fc_w = bridge.malloc(10 * FC_IN * 4)
    d_fc_b = bridge.malloc(10 * 4)
    bridge.h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
    bridge.h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
    bridge.h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    
    print("Forward pass...")
    
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    d_conv_raw = bridge.malloc(OC * BATCH * outH * outW * 4)
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    bridge.lib.gemm_forward(d_w_conv, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
    bridge.lib.apply_relu(d_conv_raw, OC * BATCH * outH * outW)
    
    d_conv = bridge.malloc(OC * BATCH * outH * outW * 4)
    bridge.lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)
    
    d_pool = bridge.malloc(OC * BATCH * 15 * 15 * 4)
    bridge.lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)
    
    h_pool = np.zeros((BATCH, FC_IN), dtype=np.float32)
    bridge.d2h(h_pool.ctypes.data, d_pool, BATCH * FC_IN * 4)
    print(f"Pool output: min={h_pool.min():.4f}, max={h_pool.max():.4f}")
    
    d_fc_in = bridge.malloc(BATCH * FC_IN * 4)
    bridge.h2d(d_fc_in, h_pool.ctypes.data, BATCH * FC_IN * 4)
    
    d_fc_out = bridge.malloc(BATCH * 10 * 4)
    bridge.lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
    
    h_out = np.zeros((BATCH, 10), dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
    print(f"FC output: min={h_out.min():.4f}, max={h_out.max():.4f}")
    
    if np.isnan(h_out).any():
        print("NaN detected in FC output!")
        return
    
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    exp_out = np.exp(h_out_shifted)
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    print(f"Loss: {loss:.4f}")
    
    print("\nBackward pass...")
    
    labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
    labels_onehot[np.arange(BATCH), y] = 1.0
    d_loss = probs - labels_onehot
    
    grad_fc_w = d_loss.T @ h_pool
    grad_fc_w = np.clip(grad_fc_w, -0.1, 0.1).flatten().astype(np.float32)
    print(f"FC grad: min={grad_fc_w.min():.4f}, max={grad_fc_w.max():.4f}")
    
    d_fc_grad_w = bridge.malloc(10 * FC_IN * 4)
    bridge.h2d(d_fc_grad_w, grad_fc_w.ctypes.data, 10 * FC_IN * 4)
    bridge.lib.apply_sgd_update(d_fc_w, d_fc_grad_w, 0.01, 10 * FC_IN)
    
    fc_w_reshaped = fc_w.reshape(10, FC_IN)
    grad_pool = d_loss @ fc_w_reshaped
    grad_pool = np.clip(grad_pool, -0.1, 0.1).astype(np.float32)
    print(f"Pool grad: min={grad_pool.min():.4f}, max={grad_pool.max():.4f}")
    
    d_pool_grad = bridge.malloc(OC * BATCH * 15 * 15 * 4)
    bridge.h2d(d_pool_grad, grad_pool.flatten().ctypes.data, OC * BATCH * 15 * 15 * 4)
    
    d_conv_grad = bridge.malloc(OC * BATCH * outH * outW * 4)
    bridge.lib.maxpool_backward_nchw(d_pool_grad, d_conv, d_conv_grad, BATCH, OC, outH, outW)
    print(f"Conv grad after pool: min={np.abs(d_conv_grad):.4f}")
    
    bridge.lib.apply_relu_backward(d_conv, d_conv_grad, OC * BATCH * outH * outW)
    
    d_conv_raw_grad = bridge.malloc(OC * BATCH * outH * outW * 4)
    bridge.lib.reorganize_backward(d_conv_grad, d_conv_raw_grad, BATCH, OC, outH, outW)
    
    d_conv_grad_w = bridge.malloc(OC * C * KH * KW * 4)
    bridge.lib.conv_backward(d_conv_raw_grad, d_x, d_w_conv, d_conv_grad_w,
                            BATCH, C, H, W, KH, KW, outH, outW, OC)
    
    bridge.lib.apply_sgd_update(d_w_conv, d_conv_grad_w, 0.01, OC * C * KH * KW)
    
    print("Backward done!")
    
    bridge.free(d_x)
    bridge.free(d_col)
    bridge.free(d_conv_raw)
    bridge.free(d_conv)
    bridge.free(d_pool)
    bridge.free(d_fc_in)
    bridge.free(d_fc_out)
    bridge.free(d_fc_grad_w)
    bridge.free(d_pool_grad)
    bridge.free(d_conv_grad)
    bridge.free(d_conv_raw_grad)
    bridge.free(d_conv_grad_w)
    bridge.free(d_w_conv)
    bridge.free(d_fc_w)
    bridge.free(d_fc_b)
    
    print("Done!")


if __name__ == "__main__":
    test()
