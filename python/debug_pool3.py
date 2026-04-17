#!/usr/bin/env python3
"""Debug actual pool output with proper dimensions"""
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
    
    with open(f"{data_root}/data_batch_1", 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x = batch[b'data'][:8].astype(np.float32) / 255.0
    x = x.reshape(-1, 3, 32, 32)
    
    BATCH, C, H, W = 8, 3, 32, 32
    KH, KW, OC = 3, 3, 32
    outH, outW = 30, 30
    
    w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    d_w = bridge.malloc(OC * C * KH * KW * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    bridge.h2d(d_w, w_conv.ctypes.data, OC * C * KH * KW * 4)
    
    # im2col
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    
    h_col = np.zeros((C * KH * KW, BATCH * outH * outW), dtype=np.float32)
    bridge.d2h(h_col.ctypes.data, d_col, col_size * 4)
    print(f"im2col: min={h_col.min():.4f}, max={h_col.max():.4f}")
    print(f"im2col col_sum[0]: {h_col[:, 0].sum():.4f}")
    
    # GEMM - output should be (OC, BATCH*outH*outW)
    d_conv_out = bridge.malloc(OC * BATCH * outH * outW * 4)
    bridge.lib.gemm_forward(d_w, d_col, d_conv_out, OC, BATCH * outH * outW, C * KH * KW)
    
    h_conv = np.zeros((OC * BATCH * outH * outW,), dtype=np.float32)
    bridge.d2h(h_conv.ctypes.data, d_conv_out, OC * BATCH * outH * outW * 4)
    print(f"gemm: min={h_conv.min():.4f}, max={h_conv.max():.4f}")
    
    # ReLU
    bridge.lib.apply_relu(d_conv_out, OC * BATCH * outH * outW * 4)
    bridge.d2h(h_conv.ctypes.data, d_conv_out, OC * BATCH * outH * outW * 4)
    print(f"relu: min={h_conv.min():.4f}, max={h_conv.max():.4f}")
    print(f"relu[0]: {h_conv[0]}, relu[1]: {h_conv[1]}")
    
    # Pool
    d_pool_out = bridge.malloc(OC * BATCH * 15 * 15 * 4)
    bridge.lib.apply_maxpool(d_conv_out, d_pool_out, BATCH, OC, outH, outW)
    
    h_pool = np.zeros((BATCH, OC * 15 * 15), dtype=np.float32)
    bridge.d2h(h_pool.ctypes.data, d_pool_out, BATCH * OC * 15 * 15 * 4)
    print(f"pool: min={h_pool.min():.4f}, max={h_pool.max():.4f}")
    print(f"pool[0,:10]: {h_pool[0,:10]}")
    print(f"pool[0,0]: {h_pool[0,0]}, pool[0,1]: {h_pool[0,1]}")
    
    # Check FC input
    print(f"\nFC_IN = {OC * 15 * 15}")
    print(f"h_pool shape = {h_pool.shape}")
    
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_col)
    bridge.free(d_conv_out)
    bridge.free(d_pool_out)
    print("Done!")


if __name__ == '__main__':
    test()
