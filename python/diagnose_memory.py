import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import os

class CudaCNNBridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        
        # Forward pass setup
        self.lib.full_forward_pass.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        # Optimizer setup
        self.lib.apply_sgd_update.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int
        ]

    def run_forward(self, N, C, H, W, KH, KW, OUT_C):
        h_input = np.random.randn(N * C * H * W).astype(np.float32)
        h_weights = np.random.randn(OUT_C * C * KH * KW).astype(np.float32)
        outH, outW = H - KH + 1, W - KW + 1
        h_output = np.zeros((OUT_C * N * (outH // 2) * (outW // 2),), dtype=np.float32)
        
        self.lib.full_forward_pass(
            h_input.ctypes.data, h_weights.ctypes.data, h_output.ctypes.data,
            N, C, H, W, KH, KW, OUT_C
        )
        return h_input, h_weights, h_output

    def update_weights(self, weights, grad, lr):
        # Move weights and grad to numpy arrays (simulating GPU data)
        w_np = np.array(weights, dtype=np.float32)
        g_np = np.array(grad, dtype=np.float32)
        
        # In a real scenario, these pointers would be GPU addresses.
        # Since we can't actually call apply_sgd_update on host pointers 
        # without a wrapper, this is where we'd check if the C++ side
        # is corrupting the memory.
        pass

def pytorch_forward(input_np, weights_np, N, C, H, W, KH, KW, OUT_C):
    x = torch.from_numpy(input_np).view(N, C, H, W)
    w = torch.from_numpy(weights_np).view(OUT_C, C, KH, KW)
    out = F.conv2d(x, w)
    out = F.relu(out)
    out = F.max_pool2d(out, kernel_size=2)
    return out.detach().numpy().flatten()

def diagnose_memory_corruption(weights_before, weights_after, grad):
    # Check if weights_after actually matches weights_before - lr * grad
    # Or if weights_after is just a copy of grad (the suspected bug)
    diff_from_grad = np.abs(weights_after - grad).max()
    if diff_from_grad < 1e-5:
        print("🚨 CORRUPTION DETECTED: Weights are EXACTLY equal to Gradients!")
        return True
    return False

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.abspath(os.path.join(base_dir, "..", "cpp", "libminimal_cuda_cnn.so"))
    bridge = CudaCNNBridge(so_path)
    
    print("--- Step 1: Forward Pass Verification ---")
    N, C, H, W, KH, KW, OUT_C = 1, 1, 8, 8, 3, 3, 2
    h_in, h_w, h_res_cuda = bridge.run_forward(N, C, H, W, KH, KW, OUT_C)
    h_res_torch = pytorch_forward(h_in, h_w, N, C, H, W, KH, KW, OUT_C)
    diff = np.abs(h_res_cuda - h_res_torch).max()
    print(f"Max Difference: {diff}")
    print("Forward Pass: SUCCESS" if diff < 1e-4 else "Forward Pass: FAILED")

    print("\n--- Step 2: Memory Corruption Simulation ---")
    # We simulate what we saw in the logs: Weights suddenly become Gradients
    sim_weights_before = np.random.randn(10).astype(np.float32)
    sim_grad = np.random.randn(10).astype(np.float32)
    
    # Scenario A: Correct Update
    lr = 0.01
    weights_correct = sim_weights_before - lr * sim_grad
    print(f"Correct update corruption check: {diagnose_memory_corruption(sim_weights_before, weights_correct, sim_grad)}")
    
    # Scenario B: The Bug (Weights = Grad)
    weights_bugged = sim_grad.copy()
    print(f"Bugged update corruption check: {diagnose_memory_corruption(sim_weights_before, weights_bugged, sim_grad)}")
