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
    ('reorganize_forward', [c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('maxpool_forward_store', [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('maxpool_backward_use_idx', [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]),
    ('conv_backward', [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]),
]:
    getattr(lib, fn).argtypes = sig

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

def alloc(size):
    return lib.gpu_malloc(size * 4)

np.random.seed(123)
std = 0.05
x_np = np.random.randn(BATCH, C1_IN, H, W).astype(np.float32) * 0.1
w_conv1_np = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * std
w_conv2_np = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * std
fc_w_np = np.random.randn(10*FC_IN).astype(np.float32) * std
fc_b_np = np.zeros(10, dtype=np.float32)

import torch
import torch.nn.functional as F
torch.manual_seed(123)
w_conv1_t = torch.from_numpy(w_conv1_np.reshape(C1_OUT, C1_IN, KH1, KW1).copy()).float().requires_grad_(True)
w_conv2_t = torch.from_numpy(w_conv2_np.reshape(C2_OUT, C2_IN, KH2, KW2).copy()).float().requires_grad_(True)
x_t = torch.from_numpy(x_np.copy()).float()

# Forward
c1 = F.conv2d(x_t, w_conv1_t); r1 = F.leaky_relu(c1, 0.1); p1 = F.max_pool2d(r1, 2)
c2 = F.conv2d(p1, w_conv2_t); r2 = F.leaky_relu(c2, 0.1); p2 = F.max_pool2d(r2, 2)
p2f = p2.view(BATCH, -1); fc_w_t = torch.from_numpy(fc_w_np.reshape(10, FC_IN).copy()).float().requires_grad_(True)
fc_b_t = torch.from_numpy(fc_b_np.copy()).float().requires_grad_(True)
f = F.linear(p2f, fc_w_t, fc_b_t)
loss = F.cross_entropy(f, torch.zeros(BATCH, dtype=torch.long)); loss.backward()

pt_w2_grad = w_conv2_t.grad.flatten().numpy().copy()
pt_w1_grad = w_conv1_t.grad.flatten().numpy().copy()
print(f"PyTorch w_conv2 grad norm: {np.linalg.norm(pt_w2_grad):.6f}")
print(f"PyTorch w_conv1 grad norm: {np.linalg.norm(pt_w1_grad):.6f}")

# === Our forward (NCHW throughout) ===
d_x = alloc(BATCH*C1_IN*H*W); d_wc1 = alloc(C1_OUT*C1_IN*KH1*KW1)
d_col1 = alloc(C1_IN*KH1*KW1*BATCH*outH1*outW1); d_c1r = alloc(C1_OUT*BATCH*outH1*outW1)
d_c1 = alloc(C1_OUT*BATCH*outH1*outW1); d_p1 = alloc(C1_OUT*BATCH*poolH1*poolW1)
d_mi1 = alloc(C1_OUT*BATCH*poolH1*poolW1)
d_col2 = alloc(C2_IN*KH2*KW2*BATCH*outH2*outW2); d_c2r = alloc(C2_OUT*BATCH*outH2*outW2)
d_c2 = alloc(C2_OUT*BATCH*outH2*outW2); d_p2 = alloc(C2_OUT*BATCH*poolH2*poolW2)
d_mi2 = alloc(C2_OUT*BATCH*poolH2*poolW2); d_wc2 = alloc(C2_OUT*C2_IN*KH2*KW2)

lib.gpu_memcpy_h2d(d_x, x_np.ctypes.data, BATCH*C1_IN*H*W*4)
lib.gpu_memcpy_h2d(d_wc1, w_conv1_np.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
lib.gpu_memcpy_h2d(d_wc2, w_conv2_np.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)

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

# FC forward
h_p2 = g2h(d_p2, BATCH*FC_IN).reshape(BATCH, FC_IN)
d_fcw = alloc(10*FC_IN); d_fcb = alloc(10); d_fi = alloc(BATCH*FC_IN); d_fo = alloc(BATCH*10)
lib.gpu_memcpy_h2d(d_fcw, fc_w_np.ctypes.data, 10*FC_IN*4)
lib.gpu_memcpy_h2d(d_fcb, fc_b_np.ctypes.data, 10*4)
lib.gpu_memcpy_h2d(d_fi, h_p2.ctypes.data, BATCH*FC_IN*4)
lib.dense_forward(d_fi, d_fcw, d_fcb, d_fo, BATCH, FC_IN, 10)
h_fc_out = g2h(d_fo, BATCH*10).reshape(BATCH, 10)

probs = np.exp(h_fc_out - h_fc_out.max(axis=1, keepdims=True))
probs = probs / probs.sum(axis=1, keepdims=True)
y = np.zeros(BATCH, dtype=np.int64)
d_loss = probs.copy(); d_loss[np.arange(BATCH), y] -= 1.0
grad_fc_w = (d_loss.T @ h_p2) / BATCH
grad_pool2 = (d_loss @ fc_w_np.reshape(10, FC_IN)) / BATCH

# FC backward
fc_w_t2 = torch.from_numpy(fc_w_np.reshape(10, FC_IN).copy()).float().requires_grad_(True)
fc_b_t2 = torch.from_numpy(fc_b_np.copy()).float().requires_grad_(True)
p2_t = torch.from_numpy(h_p2.copy()).float().requires_grad_(True)
f2 = F.linear(p2_t, fc_w_t2, fc_b_t2)
loss2 = F.cross_entropy(f2, torch.zeros(BATCH, dtype=torch.long)); loss2.backward()
pt_fc_w_grad = fc_w_t2.grad.flatten().numpy()

# CONV2 backward: use actual stored indices
# d_p2g: grad w.r.t pool2 output, shape (BATCH, C2_OUT, poolH2, poolW2) NCHW
d_p2g = alloc(C2_OUT*BATCH*poolH2*poolW2)
grad_p2_clip = np.clip(grad_pool2.reshape(BATCH, C2_OUT, poolH2, poolW2), -1.0, 1.0)
lib.gpu_memcpy_h2d(d_p2g, grad_p2_clip.flatten().astype(np.float32).ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)

# d_c2gr: grad after maxpool_backward, shape (BATCH, C2_OUT, outH2, outW2) NCHW
d_c2gr = alloc(C2_OUT*BATCH*outH2*outW2); lib.gpu_memset(d_c2gr, 0, C2_OUT*BATCH*outH2*outW2*4)
lib.maxpool_backward_use_idx(d_p2g, d_mi2, d_c2gr, BATCH, C2_OUT, outH2, outW2)

h_c2gr = g2h(d_c2gr, C2_OUT*BATCH*outH2*outW2).reshape(C2_OUT, BATCH, outH2, outW2)
print(f"\nOur d_c2gr (NCHW): min={h_c2gr.min():.6f}, max={h_c2gr.max():.6f}")
print(f"PyTorch r2 (NCHW): min={r2.min():.6f}, max={r2.max():.6f}")

# Compute conv2 weight gradient using our system
d_wc2g = alloc(C2_OUT*C2_IN*KH2*KW2); d_p1g = alloc(BATCH*C2_IN*poolH1*poolW1)
lib.gpu_memset(d_wc2g, 0, C2_OUT*C2_IN*KH2*KW2*4); lib.gpu_memset(d_p1g, 0, BATCH*C2_IN*poolH1*poolW1*4)

# conv_backward(grad_out, input, weights, grad_weights, grad_input, ...)
# For CONV2: N=BATCH, C=C2_IN=32, H=poolH1=15, W=poolW1=15, KH=KH2=3, KW=KW2=3
# outH=outH2=13, outW=outW2=13, OUT_C=C2_OUT=64
lib.conv_backward(d_c2gr, d_p1, d_wc2, d_wc2g, d_p1g, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
h_wc2g = g2h(d_wc2g, C2_OUT*C2_IN*KH2*KW2).reshape(-1) / BATCH
print(f"\nCONV2 weight grad norm: ours={np.linalg.norm(h_wc2g):.6f}, pt={np.linalg.norm(pt_w2_grad):.6f}")
print(f"Ratio: {np.linalg.norm(h_wc2g)/np.linalg.norm(pt_w2_grad):.4f}")

# CONV1 backward
d_c1gr = alloc(C1_OUT*BATCH*outH1*outW1); lib.gpu_memset(d_c1gr, 0, C1_OUT*BATCH*outH1*outW1*4)
lib.maxpool_backward_use_idx(d_p1g, d_mi1, d_c1gr, BATCH, C1_OUT, outH1, outW1)
d_wc1g = alloc(C1_OUT*C1_IN*KH1*KW1); d_xg = alloc(BATCH*C1_IN*H*W)
lib.gpu_memset(d_wc1g, 0, C1_OUT*C1_IN*KH1*KW1*4); lib.gpu_memset(d_xg, 0, BATCH*C1_IN*H*W*4)

lib.conv_backward(d_c1gr, d_x, d_wc1, d_wc1g, d_xg, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
h_wc1g = g2h(d_wc1g, C1_OUT*C1_IN*KH1*KW1).reshape(-1) / BATCH
print(f"\nCONV1 weight grad norm: ours={np.linalg.norm(h_wc1g):.6f}, pt={np.linalg.norm(pt_w1_grad):.6f}")
print(f"Ratio: {np.linalg.norm(h_wc1g)/np.linalg.norm(pt_w1_grad):.4f}")

# Print some sample gradients to compare
print(f"\nPyTorch w_conv2[0,:5]: {pt_w2_grad[:5]}")
print(f"Ours w_conv2[0,:5]: {h_wc2g[:5]}")
print(f"PyTorch w_conv1[0,:5]: {pt_w1_grad[:5]}")
print(f"Ours w_conv1[0,:5]: {h_wc1g[:5]}")

for p in [d_x,d_wc1,d_col1,d_c1r,d_c1,d_p1,d_mi1,d_col2,d_c2r,d_c2,d_p2,d_mi2,d_fcw,d_fcb,d_fi,d_fo,d_wc2,d_p2g,d_c2gr,d_wc2g,d_p1g,d_c1gr,d_wc1g,d_xg]:
    lib.gpu_free(p)
