#include "cuda_check.h"

// Layout conversion kernels for NCHW <-> CNHW
// NCHW: (N, C, H, W) flat = n*C*H*W + c*H*W + h*W + w
// CNHW: (C, N, H, W) flat = c*N*H*W + n*H*W + h*W + w

// NCHW -> CNHW
__global__ void nchw_to_cnhw_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // input NCHW flat idx -> output CNHW flat idx
    // output[c,n,h,w] = input[n,c,h,w]
    int out_idx = c * (N * H * W) + n * H * W + h * W + w;
    output[out_idx] = input[idx];
}

// CNHW -> NCHW
__global__ void cnhw_to_nchw_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = (idx / (W * H)) % N;
    int c = idx / (W * H * N);

    // output[n,c,h,w] = input[c,n,h,w]
    int out_idx = n * (C * H * W) + c * H * W + h * W + w;
    output[out_idx] = input[idx];
}

extern "C" {
    void nchw_to_cnhw(float* d_input, float* d_output, int N, int C, int H, int W) {
        int total = N * C * H * W;
        int tpb = 256;
        nchw_to_cnhw_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_input, d_output, N, C, H, W);
        CUDA_KERNEL_CHECK();
    }

    void cnhw_to_nchw(float* d_input, float* d_output, int N, int C, int H, int W) {
        int total = N * C * H * W;
        int tpb = 256;
        cnhw_to_nchw_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_input, d_output, N, C, H, W);
        CUDA_KERNEL_CHECK();
    }
}
