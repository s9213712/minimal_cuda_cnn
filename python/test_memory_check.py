#!/usr/bin/env python3
"""Memory-optimized training with pre-allocated buffers"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os, pickle, time, gc

workspace = "/mnt/c/Users/user/.openclaw/workspace"
so = os.path.join(workspace, "NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so")
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
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

BATCH = 64
LR_CONV = 0.01
LR_CONV1 = 0.001
LR_FC = 0.01
EPOCHS = 10

C1_IN, C1_OUT = 3, 32; H, W = 32, 32; KH1, KW1 = 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1
poolH1, poolW1 = outH1 // 2, outW1 // 2

C2_IN, C2_OUT = 32, 64; KH2, KW2 = 3, 3
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2

FC_IN = C2_OUT * poolH2 * poolW2

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

def h2d(ptr, arr):
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)

# Memory usage per allocation
def mb(elems): return elems * 4 / 1024 / 1024

print("Buffer sizes:")
print(f"  d_x:            {mb(BATCH*C1_IN*H*W):.1f} MB")
print(f"  d_col1:         {mb(C1_IN*KH1*KW1*BATCH*outH1*outW1):.1f} MB")
print(f"  d_conv1_raw:    {mb(C1_OUT*BATCH*outH1*outW1):.1f} MB")
print(f"  d_conv1:        {mb(C1_OUT*BATCH*outH1*outW1):.1f} MB")
print(f"  d_pool1:        {mb(C1_OUT*BATCH*poolH1*poolW1):.1f} MB")
print(f"  d_max_idx1:     {mb(C1_OUT*BATCH*poolH1*poolW1):.1f} MB")
print(f"  d_col2:         {mb(C2_IN*KH2*KW2*BATCH*outH2*outW2):.1f} MB")
print(f"  d_conv2_raw:    {mb(C2_OUT*BATCH*outH2*outW2):.1f} MB")
print(f"  d_conv2:        {mb(C2_OUT*BATCH*outH2*outW2):.1f} MB")
print(f"  d_pool2:        {mb(C2_OUT*BATCH*poolH2*poolW2):.1f} MB")
print(f"  d_max_idx2:     {mb(C2_OUT*BATCH*poolH2*poolW2):.1f} MB")
print(f"  d_fc_in:        {mb(BATCH*FC_IN):.1f} MB")
print(f"  d_fc_out:       {mb(BATCH*10):.1f} MB")
print(f"  d_pool2_grad:   {mb(C2_OUT*BATCH*poolH2*poolW2):.1f} MB")
print(f"  d_conv2_grad_raw: {mb(C2_OUT*BATCH*outH2*outW2):.1f} MB")
print(f"  d_conv2_grad:   {mb(C2_OUT*BATCH*outH2*outW2):.1f} MB")
print(f"  d_w_conv2_grad: {mb(C2_OUT*C2_IN*KH2*KW2):.1f} MB")
print(f"  d_pool1_grad:  {mb(C2_IN*BATCH*poolH1*poolW1):.1f} MB")
print(f"  d_conv1_grad_raw: {mb(C1_OUT*BATCH*outH1*outW1):.1f} MB")
print(f"  d_conv1_grad:   {mb(C1_OUT*BATCH*outH1*outW1):.1f} MB")
print(f"  d_w_conv1_grad: {mb(C1_OUT*C1_IN*KH1*KW1):.1f} MB")
print(f"  d_x_grad:      {mb(BATCH*C1_IN*H*W):.1f} MB")
print(f"  h_pool2:        {mb(BATCH*FC_IN):.1f} MB (CPU)")
print(f"  d_fc_grad_w:    {mb(10*FC_IN):.1f} MB")
total_gpu = (BATCH*C1_IN*H*W + C1_IN*KH1*KW1*BATCH*outH1*outW1 + C1_OUT*BATCH*outH1*outW1*2 +
             C1_OUT*BATCH*poolH1*poolW1*2 + C2_IN*KH2*KW2*BATCH*outH2*outW2 + C2_OUT*BATCH*outH2*outW2*2 +
             C2_OUT*BATCH*poolH2*poolW2*2 + BATCH*FC_IN + BATCH*10 + C2_OUT*C2_IN*KH2*KW2 +
             C2_IN*BATCH*poolH1*poolW1 + C1_OUT*C1_IN*KH1*KW1 + BATCH*C1_IN*H*W + 10*FC_IN)
print(f"\nTotal GPU allocation (with 2x for some buffers): ~{mb(total_gpu):.0f} MB")
print(f"RTX 3050 has 4GB = {4096:.0f} MB")
print(f"Margin: {4096 - mb(total_gpu):.0f} MB")
