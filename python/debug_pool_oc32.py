#!/usr/bin/env python3
"""Debug pool with OC=32 like original"""
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
        self.lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


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
    outH, outW = H - KH + 1, W - KW + 1
    
    np.random.seed(42)
    w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    d_w = bridge.malloc(OC * C * KH * KW * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    bridge.h2d(d_w, w_conv.ctypes.data, OC * C * KH * KW * 4)
    
    # Conv
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    d_conv_out = bridge.malloc(OC * BATCH * outH * outW * 4)
    
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    bridge.lib.gemm_forward(d_w, d_col, d_conv_out, OC, BATCH * outH * outW, C * KH * KW)
    bridge.lib.apply_relu(d_conv_out, OC * BATCH * outH * outW * 4)
    
    # Check a few values
    h_conv = np.zeros((BATCH, OC * outH * outW), dtype=np.float32)
    bridge.d2h(h_conv.ctypes.data, d_conv_out, BATCH * OC * outH * outW * 4)
    print(f"Conv output (flattened): min={h_conv.min():.4f}, max={h_conv.max():.4f}")
    print(f"Conv output[0,0:10]: {h_conv[0,0:10]}")
    
    # Pool
    poolH, poolW = outH // 2, outW // 2
    print(f"\nPool input: {OC*BATCH*outH*outW} elements")
    print(f"Pool output: {OC*BATCH*poolH*poolW} elements")
    
    d_pool_out = bridge.malloc(OC * BATCH * poolH * poolW * 4)
    bridge.lib.apply_maxpool(d_conv_out, d_pool_out, BATCH, OC, outH, outW)
    
    h_pool = np.zeros((BATCH, OC * poolH * poolW), dtype=np.float32)
    bridge.d2h(h_pool.ctypes.data, d_pool_out, BATCH * OC * poolH * poolW * 4)
    print(f"Pool output: min={h_pool.min():.4f}, max={h_pool.max():.4f}")
    print(f"Pool output[0,0:10]: {h_pool[0,0:10]}")
    
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_col)
    bridge.free(d_conv_out)
    bridge.free(d_pool_out)
    print("Done!")


if __name__ == '__main__':
    test()
