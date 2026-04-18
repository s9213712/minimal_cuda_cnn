#include "dense_layer.h"
#include "tensor.h"
#include "cuda_check.h"
#include <cuda_runtime.h>

__global__ void dense_forward_kernel(const float* input, const float* weights, const float* bias, float* output, int N, int in_f, int out_f) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < out_f) {
        float sum = 0.0f;
        for (int i = 0; i < in_f; i++) {
            sum += input[row * in_f + i] * weights[col * in_f + i];
        }
        output[row * out_f + col] = sum + bias[col];
    }
}

// FC Backward: dL/dinput = dL/dout @ weights^T
// weights layout: (out_f × in_f) row-major: weights[col*in_f + i]
__global__ void dense_backward_input_kernel(const float* d_out, const float* weights, float* d_input,
                                             int N, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * in_f;
    if (idx >= total) return;

    int n = idx / in_f;
    int i = idx % in_f;

    float sum = 0.0f;
    // d_input[n,i] = sum_over_k(d_out[n,k] * weights[k,i])
    // weights[k,i] = weights[k * in_f + i]
    for (int k = 0; k < out_f; k++) {
        sum += d_out[n * out_f + k] * weights[k * in_f + i];
    }
    d_input[idx] = sum;
}

// FC Backward: dL/dweights = dL/dout^T @ input
// d_weights[out_f, in_f] += input[n] * d_out[n, out_f] (for each n)
__global__ void dense_backward_weights_kernel(const float* d_out, const float* input, float* d_weights,
                                                int N, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_f * in_f;
    if (idx >= total) return;

    int col = idx / in_f;  // output feature
    int i = idx % in_f;    // input feature

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += input[n * in_f + i] * d_out[n * out_f + col];
    }
    d_weights[idx] = sum;
}

// FC Backward: dL/dweights with atomicAdd for safe accumulation across thread blocks
// (Used when launching multiple thread blocks that may write to same weight)
__global__ void dense_backward_weights_atomic_kernel(const float* d_out, const float* input, float* d_weights,
                                                      int N, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_f * in_f;
    if (idx >= total) return;

    int col = idx / in_f;  // output feature
    int i = idx % in_f;    // input feature

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += input[n * in_f + i] * d_out[n * out_f + col];
    }
    // Normalize by batch and use atomicAdd for thread safety
    atomicAdd(&d_weights[idx], sum / (float)N);
}

// FC Backward: dL/dbias = sum over N of dL/dout
__global__ void dense_backward_bias_kernel(const float* d_out, float* d_bias, int N, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += d_out[n * out_f + idx];
    }
    d_bias[idx] = sum;
}

CudaTensor* DenseLayer::forward(CudaTensor* input) {
    int batch_size = input->n;
    int total_in = input->c * input->h * input->w;
    
    CudaTensor* output = new CudaTensor(batch_size, out_features, 1, 1);
    
    dim3 block(16, 16);
    dim3 grid((out_features + 15) / 16, (batch_size + 15) / 16);
    
    dense_forward_kernel<<<grid, block>>>(input->data, d_weights, d_bias, output->data, batch_size, total_in, out_features);
    CUDA_KERNEL_CHECK();
    
    return output;
}

extern "C" {
    void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output, int N, int in_f, int out_f) {
        dim3 block(16, 16);
        dim3 grid((out_f + 15) / 16, (N + 15) / 16);
        dense_forward_kernel<<<grid, block>>>(d_input, d_weights, d_bias, d_output, N, in_f, out_f);
        CUDA_KERNEL_CHECK();
    }

    // FC backward - all on GPU, no CPU roundtrip
    // d_dout: (N, out_f), input: (N, in_f), weights: (out_f, in_f)
    // outputs: d_din: (N, in_f), d_weights: (out_f, in_f), d_bias: (out_f)
    // FC backward - all on GPU, no CPU roundtrip (includes bias grad)
    // Renamed to avoid conflict with backward.cu's dense_backward
    void dense_backward_full(float* d_dout, float* d_input, float* d_weights,
                        float* d_din, float* d_dweights, float* d_dbias,
                        int N, int in_f, int out_f) {
        int tpb = 256;
        int nin = N * in_f;
        dense_backward_input_kernel<<<(nin + tpb-1) / tpb, tpb>>>(d_dout, d_weights, d_din, N, in_f, out_f);

        int nw = out_f * in_f;
        dense_backward_weights_kernel<<<(nw + tpb-1) / tpb, tpb>>>(d_dout, d_input, d_dweights, N, in_f, out_f);

        dense_backward_bias_kernel<<<(out_f + tpb-1) / tpb, tpb>>>(d_dout, d_dbias, N, out_f);

        CUDA_KERNEL_CHECK();
    }
}
