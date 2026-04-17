import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int

BATCH = 4; H, W = 32, 32
C1_IN, C1_OUT = 3, 32; KH1, KW1 = 3, 3
outH1, outW1 = H-KH1+1, W-KW1+1; poolH1, poolW1 = outH1//2, outW1//2
C2_IN, C2_OUT = 32, 64; KH2, KW2 = 3, 3
outH2, outW2 = poolH1-KH2+1, poolW1-KW2+1; poolH2, poolW2 = outH2//2, outW2//2
FC_IN = C2_OUT * poolH2 * poolW2

lib = ctypes.CDLL('/home/s92137/NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so')
lib.gpu_malloc.restype = c_void_p
for fn, sig in [
    ('gpu_malloc', [ctypes.c_size_t]), ('gpu_free', [c_void_p]),
    ('gpu_memcpy_h2d', [c_void_p, c_void_p, ctypes.c_size_t]),
    ('gpu_memcpy_d2h', [c_void_p, c_void_p, ctypes.c_size_t]),
    ('gpu_memset', [c_void_p, c_int, ctypes.c_size_t]),
    ('im2col_forward', [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]),
    ('gemm_forward', [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]),
    ('leaky_relu_forward', [c_void_p, c_float, c_int]),
    ('dense_forward', [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]),
    ('reorganize_forward', [c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('reorganize_backward', [c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('maxpool_forward_store', [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('maxpool_backward_use_idx', [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('conv_backward', [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]),
]:
    getattr(lib, fn).argtypes = sig

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

def print_layout(ptr, shape, name):
    """Print min/max/mean to help identify layout"""
    data = g2h(ptr, np.prod(shape)).reshape(shape)
    print(f"{name}: shape={shape}, min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    return data

np.random.seed(123)
std = 0.05
w_conv1_np = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * std
w_conv2_np = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * std
fc_w_np = np.random.randn(10*FC_IN).astype(np.float32) * std
fc_b_np = np.zeros(10, dtype=np.float32)
x_np = np.random.randn(BATCH, C1_IN, H, W).astype(np.float32) * 0.1

import torch
import torch.nn.functional as F
torch.manual_seed(123)
w_conv1_t = torch.from_numpy(w_conv1_np.reshape(C1_OUT, C1_IN, KH1, KW1).copy()).float().requires_grad_(True)
w_conv2_t = torch.from_numpy(w_conv2_np.reshape(C2_OUT, C2_IN, KH2, KW2).copy()).float().requires_grad_(True)
fc_w_t = torch.from_numpy(fc_w_np.reshape(10, FC_IN).copy()).float().requires_grad_(True)
fc_b_t = torch.from_numpy(fc_b_np.copy()).float().requires_grad_(True)
x_t = torch.from_numpy(x_np.copy()).float().requires_grad_(True)
c1 = F.conv2d(x_t, w_conv1_t); r1 = F.leaky_relu(c1, 0.1); p1 = F.max_pool2d(r1, 2)
c2 = F.conv2d(p1, w_conv2_t); r2 = F.leaky_relu(c2, 0.1); p2 = F.max_pool2d(r2, 2)
p2f = p2.view(BATCH, -1); f = F.linear(p2f, fc_w_t, fc_b_t)
loss = F.cross_entropy(f, torch.zeros(BATCH, dtype=torch.long)); loss.backward()
pt = {'w1':w_conv1_t.grad.flatten().numpy().copy(), 'w2':w_conv2_t.grad.flatten().numpy().copy(),
      'fc_w':fc_w_t.grad.flatten().numpy().copy(), 'x':x_t.grad.flatten().numpy().copy(),
      'c1':c1.detach().numpy(), 'r1':r1.detach().numpy(), 'p1':p1.detach().numpy(),
      'c2':c2.detach().numpy(), 'r2':r2.detach().numpy(), 'p2':p2.detach().numpy()}

def alloc(size):
    return lib.gpu_malloc(size * 4)

d_x = alloc(BATCH*C1_IN*H*W); d_wc1 = alloc(C1_OUT*C1_IN*KH1*KW1)
d_col1 = alloc(C1_IN*KH1*KW1*BATCH*outH1*outW1); d_c1r = alloc(C1_OUT*BATCH*outH1*outW1)
d_c1 = alloc(C1_OUT*BATCH*outH1*outW1); d_p1 = alloc(C1_OUT*BATCH*poolH1*poolW1)
d_mi1 = alloc(C1_OUT*BATCH*poolH1*poolW1); d_col2 = alloc(C2_IN*KH2*KW2*BATCH*outH2*outW2)
d_c2r = alloc(C2_OUT*BATCH*outH2*outW2); d_c2 = alloc(C2_OUT*BATCH*outH2*outW2)
d_p2 = alloc(C2_OUT*BATCH*poolH2*poolW2); d_mi2 = alloc(C2_OUT*BATCH*poolH2*poolW2)
d_fcw = alloc(10*FC_IN); d_fcb = alloc(10); d_fi = alloc(BATCH*FC_IN)
d_fo = alloc(BATCH*10); d_wc2 = alloc(C2_OUT*C2_IN*KH2*KW2)

lib.gpu_memcpy_h2d(d_x, x_np.ctypes.data, BATCH*C1_IN*H*W*4)
lib.gpu_memcpy_h2d(d_wc1, w_conv1_np.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
lib.gpu_memcpy_h2d(d_wc2, w_conv2_np.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
lib.gpu_memcpy_h2d(d_fcw, fc_w_np.ctypes.data, 10*FC_IN*4)
lib.gpu_memcpy_h2d(d_fcb, fc_b_np.ctypes.data, 10*4)

lib.im2col_forward(d_x, d_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
lib.gemm_forward(d_wc1, d_col1, d_c1r, C1_OUT, BATCH*outH1*outW1, C1_IN*KH1*KW1)
lib.leaky_relu_forward(d_c1r, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
lib.reorganize_forward(d_c1r, d_c1, BATCH, C1_OUT, outH1, outW1)
lib.maxpool_forward_store(d_p1, d_c1, d_mi1, BATCH, C1_OUT, outH1, outW1)

lib.im2col_forward(d_p1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
lib.gemm_forward(d_wc2, d_col2, d_c2r, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
lib.leaky_relu_forward(d_c2r, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
lib.reorganize_forward(d_c2r, d_c2, BATCH, C2_OUT, outH2, outW2)
lib.maxpool_forward_store(d_p2, d_c2, d_mi2, BATCH, C2_OUT, outH2, outW2)

# Print forward intermediate layouts
print("=== FORWARD LAYOUTS ===")
print(f"PyTorch c1 (NCHW): {pt['c1'].shape}, PyTorch r1 (NCHW): {pt['r1'].shape}")
print(f"PyTorch p1 (NCHW): {pt['p1'].shape}")
print(f"PyTorch c2 (NCHW): {pt['c2'].shape}, PyTorch r2 (NCHW): {pt['r2'].shape}")
print(f"PyTorch p2 (NCHW): {pt['p2'].shape}")

print_layout(d_c1r, (C1_OUT, BATCH*outH1*outW1), "d_c1r (gemm out)")
print_layout(d_c1, (BATCH, C1_OUT, outH1, outW1), "d_c1 (after reorganize)")
print_layout(d_p1, (BATCH, C1_OUT, poolH1, poolW1), "d_p1 (pool1 out)")
print_layout(d_c2r, (C2_OUT, BATCH*outH2*outW2), "d_c2r (gemm out)")
print_layout(d_c2, (BATCH, C2_OUT, outH2, outW2), "d_c2 (after reorganize)")
print_layout(d_p2, (BATCH, C2_OUT, poolH2, poolW2), "d_p2 (pool2 out)")

# Check d_mi2 indices
h_mi2 = g2h(d_mi2, BATCH*C2_OUT*poolH2*poolW2).reshape(BATCH, C2_OUT, poolH2, poolW2)
h_c2 = pt['c2']  # NCHW
# Find max position for batch=0, channel=0
b, c = 0, 0
hw_idx = np.argmax(h_p2[0, 0].flatten()) if b == 0 else 0
print(f"\nd_mi2 sample[0,0,:3,:3]:\n{h_mi2[0,0,:3,:3].astype(int)}")
print(f"PyTorch c2[0,0,:3,:3]:\n{h_c2[0,0,:3,:3]}")
print(f"PyTorch p2[0,0,:3,:3]:\n{pt['p2'][0,0,:3,:3]}")

# FC
h_p2_cpu = g2h(d_p2, BATCH*FC_IN).reshape(BATCH, FC_IN)
lib.gpu_memcpy_h2d(d_fi, h_p2_cpu.ctypes.data, BATCH*FC_IN*4)
lib.dense_forward(d_fi, d_fcw, d_fcb, d_fo, BATCH, FC_IN, 10)
h_fc_out = g2h(d_fo, BATCH*10).reshape(BATCH, 10)

probs = np.exp(h_fc_out - h_fc_out.max(axis=1, keepdims=True))
probs = probs / probs.sum(axis=1, keepdims=True)
y = np.zeros(BATCH, dtype=np.int64)
d_loss = probs.copy(); d_loss[np.arange(BATCH), y] -= 1.0
grad_fc_w = (d_loss.T @ h_p2_cpu) / BATCH
grad_pool2 = (d_loss @ fc_w_np.reshape(10, FC_IN)) / BATCH

print(f"\nFC grad: ours={np.linalg.norm(grad_fc_w):.6f}, pt={np.linalg.norm(pt['fc_w']):.6f}")

# BACKWARD CHECK - verify what conv_backward receives
print("\n=== BACKWARD: CONV2 ===")
d_p2g = alloc(C2_OUT*BATCH*poolH2*poolW2)
grad_p2_clip = np.clip(grad_pool2.reshape(BATCH, C2_OUT, poolH2, poolW2), -1.0, 1.0)
lib.gpu_memcpy_h2d(d_p2g, grad_p2_clip.flatten().astype(np.float32).ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)
d_c2gr = alloc(C2_OUT*BATCH*outH2*outW2); lib.gpu_memset(d_c2gr, 0, C2_OUT*BATCH*outH2*outW2*4)
lib.maxpool_backward_use_idx(d_p2g, d_mi2, d_c2gr, BATCH, C2_OUT, outH2, outW2)
print_layout(d_c2gr, (BATCH, C2_OUT, outH2, outW2), "d_c2gr (after maxpool_backward)")
# PyTorch r2 has shape (BATCH, C2_OUT, outH2, outW2)
print(f"PyTorch r2 (leaky_relu output, NCHW): {pt['r2'].shape}")
print(f"PyTorch r2[0,0,:3,:3]:\n{pt['r2'][0,0,:3,:3]}")

for p in [d_x,d_wc1,d_col1,d_c1r,d_c1,d_p1,d_mi1,d_col2,d_c2r,d_c2,d_p2,d_mi2,d_fcw,d_fcb,d_fi,d_fo,d_wc2,d_p2g,d_c2gr]:
    lib.gpu_free(p)
