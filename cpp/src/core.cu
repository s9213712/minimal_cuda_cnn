#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// CUDA Kernels
// -----------------------------------------------------------------------------

__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void maxpool_forward_kernel(const float* input, float* output, int n, int c, int h, int w) {
    int out_h = h / 2; int out_w = w / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * c * out_h * out_w;
    if (idx >= total_elements) return;
    int ow = idx % out_w; int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % c; int on = idx / (out_w * out_h * c);
    int in_h_start = oh * 2; int in_w_start = ow * 2;
    float max_val = -1e38f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int in_idx = ((on * c + oc) * h + (in_h_start + i)) * w + (in_w_start + j);
            max_val = fmaxf(max_val, input[in_idx]);
        }
    }
    output[idx] = max_val;
}

__global__ void im2col_kernel(const float* input, float* output, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = (C * KH * KW) * (N * outH * outW);
    if (idx >= total_elements) return;
    int row = idx / (N * outH * outW); 
    int col = idx % (N * outH * outW); 
    int c = row / (KH * KW); int kh = (row / KW) % KH; int kw = row % KW;
    int n = col / (outH * outW); int ow = col % outW; int oh = (col / outW) % outH;
    output[idx] = input[((n * C + c) * H + (oh + kh)) * W + (ow + kw)];
}

__global__ void gemm_kernel(const float* A, const float* B, float* C_out, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
        C_out[row * N + col] = sum;
    }
}

extern "C" {
    void im2col_forward(float* d_input, float* d_output, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
        int total_elements = (C * KH * KW) * (N * outH * outW);
        int tpb = 256;
        im2col_kernel<<<(total_elements + tpb - 1) / tpb, tpb>>>(d_input, d_output, N, C, H, W, KH, KW, outH, outW);
        cudaDeviceSynchronize();
    }

    void gemm_forward(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    void apply_relu(float* d_data, int size) {
        int tpb = 256;
        relu_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        cudaDeviceSynchronize();
    }

    void apply_maxpool(float* d_input, float* d_output, int n, int c, int h, int w) {
        int out_h = h / 2; int out_w = w / 2;
        int size = n * c * out_h * out_w;
        int tpb = 256;
        maxpool_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_input, d_output, n, c, h, w);
        cudaDeviceSynchronize();
    }

    // Layout conversion functions defined in layout_convert.cu
    void nchw_to_cnhw(float* d_input, float* d_output, int N, int C, int H, int W);
    void cnhw_to_nchw(float* d_input, float* d_output, int N, int C, int H, int W);

    // Forward declarations for functions implemented in other .cu files
    void reorganize_forward(float* d_input, float* d_output, int N, int C, int H, int W);
    void reorganize_backward(float* d_grad_output, float* d_grad_input, int N, int C, int H, int W);
    void layer_norm_forward(float* d_output, float* d_input, float* d_gamma, float* d_beta, int N, int C, int H, int W, float eps);
    void layer_norm_backward(float* d_grad_input, float* d_grad_output, float* d_input, float* d_gamma, int N, int C, int H, int W, float eps);
    void maxpool_forward_store(float* d_output, float* d_input, int* d_max_idx, int N, int C, int H, int W);
    void maxpool_backward_use_idx(float* d_grad_out, int* d_max_idx, float* d_grad_input, int N, int C, int H, int W);
}
