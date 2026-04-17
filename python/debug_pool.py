#!/usr/bin/env python3
"""Debug conv+pool output"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int
import pickle

class Bridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = c_void_p
        self.lib.gpu_free.argtypes = [c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    bridge = Bridge(so_path)
    
    # Load a batch
    with open(f"{data_root}/data_batch_1", 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x = batch[b'data'][:8].astype(np.float32) / 255.0
    
    x = x.reshape(-1, 3, 32, 32)
    
    BATCH, C, H, W = 8, 3, 32, 32
    KH, KW, OC = 3, 3, 32
    
    # Init conv weights
    w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    d_w_conv = bridge.malloc(OC * C * KH * KW * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    bridge.h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
    
    # Conv
    outH, outW = 30, 30
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    d_conv_out = bridge.malloc(OC * BATCH * outH * outW * 4)
    
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    bridge.lib.gemm_forward(d_w_conv, d_col, d_conv_out, OC, BATCH * outH * outW, C * KH * KW)
    bridge.lib.apply_relu(d_conv_out, OC * BATCH * outH * outW)
    
    h_conv_out = np.zeros((OC * BATCH * outH * outW,), dtype=np.float32)
    bridge.d2h(h_conv_out.ctypes.data, d_conv_out, OC * BATCH * outH * outW * 4)
    print(f"Conv out: min={h_conv_out.min():.4f}, max={h_conv_out.max():.4f}, anyNaN={np.isnan(h_conv_out).any()}")
    
    # Pool
    d_pool_out = bridge.malloc(OC * BATCH * 15 * 15 * 4)
    bridge.lib.apply_maxpool(d_conv_out, d_pool_out, BATCH, OC, outH, outW)
    
    h_pool = np.zeros((BATCH, OC * 15 * 15), dtype=np.float32)
    bridge.d2h(h_pool.ctypes.data, d_pool_out, BATCH * OC * 15 * 15 * 4)
    print(f"Pool out: min={h_pool.min():.4f}, max={h_pool.max():.4f}, anyNaN={np.isnan(h_pool).any()}")
    
    bridge.free(d_x)
    bridge.free(d_w_conv)
    bridge.free(d_col)
    bridge.free(d_conv_out)
    bridge.free(d_pool_out)
    
    print("Done!")


if __name__ == '__main__':
    test()
