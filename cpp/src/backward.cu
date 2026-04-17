#include <cuda_runtime.h>
#include <cmath>

// ============== Activation Backward ==============
__global__ void relu_backward_kernel(float* data, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (data[idx] > 0) ? grad[idx] : 0.0f;
    }
}

__global__ void relu_backward_inplace_kernel(float* data_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data_grad[idx] = (data_grad[idx] > 0) ? data_grad[idx] : 0.0f;
    }
}

// ============== MaxPool Backward ==============
__global__ void maxpool_backward_kernel(const float* grad_out, const float* input, float* grad_input,
                                        int n, int c, int h, int w) {
    int out_h = h / 2;
    int out_w = w / 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * out_h * out_w;
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % c;
    int on = idx / (out_w * out_h * c);
    
    int h_start = oh * 2;
    int w_start = ow * 2;
    
    // Find which input element was the max
    float max_val = -1e38f;
    int max_h = h_start, max_w = w_start;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int in_idx = ((on * c + oc) * h + (h_start + i)) * w + (w_start + j);
            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_h = h_start + i;
                max_w = w_start + j;
            }
        }
    }
    
    // Propagate gradient only to max location
    int grad_in_idx = ((on * c + oc) * h + max_h) * w + max_w;
    int grad_out_idx = idx;
    atomicAdd(&grad_input[grad_in_idx], grad_out[grad_out_idx]);
}

extern "C" {
    void apply_relu_backward(float* data, float* grad, int size) {
        int tpb = 256;
        relu_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, size);
        cudaDeviceSynchronize();
    }
    
    void maxpool_backward(float* d_grad_out, float* d_input, float* d_grad_input, int n, int c, int h, int w) {
        int out_h = h / 2;
        int out_w = w / 2;
        int size = n * c * out_h * out_w;
        int tpb = 256;
        maxpool_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_grad_out, d_input, d_grad_input, n, c, h, w);
        cudaDeviceSynchronize();
    }
}
