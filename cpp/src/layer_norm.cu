#include "cuda_check.h"

// Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Input: (N, C, H, W) -> normalize over (H*W)
// Output: (N, C, H, W)

__global__ void layer_norm_forward_kernel(float* output, const float* input,
                                           const float* gamma, const float* beta,
                                           int N, int C, int H, int W, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // Compute mean over H*W for this (n, c)
    float mean = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int in_idx = ((n * C + c) * H + i) * W + j;
            mean += input[in_idx];
        }
    }
    mean /= (H * W);

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int in_idx = ((n * C + c) * H + i) * W + j;
            float diff = input[in_idx] - mean;
            var += diff * diff;
        }
    }
    var /= (H * W);

    // Normalize
    float inv_std = rsqrtf(var + eps);
    int out_idx = ((n * C + c) * H + h) * W + w;
    output[out_idx] = ((input[out_idx] - mean) * inv_std) * gamma[c] + beta[c];
}

__global__ void layer_norm_backward_kernel(float* grad_input, float* grad_output,
                                            const float* input, const float* gamma,
                                            int N, int C, int H, int W, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // Compute mean over H*W for this (n, c)
    float mean = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int in_idx = ((n * C + c) * H + i) * W + j;
            mean += input[in_idx];
        }
    }
    mean /= (H * W);

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int in_idx = ((n * C + c) * H + i) * W + j;
            float diff = input[in_idx] - mean;
            var += diff * diff;
        }
    }
    var /= (H * W);

    float inv_std = rsqrtf(var + eps);
    float x_hat = (input[idx] - mean) * inv_std;

    // dL/dx = (gamma * inv_std) * (dL/dy - x_hat * sum(dL/dy * x_hat))
    float sum_dy_xhat = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int out_idx = ((n * C + c) * H + i) * W + j;
            float x_hat_j = (input[out_idx] - mean) * inv_std;
            sum_dy_xhat += grad_output[out_idx] * x_hat_j;
        }
    }

    int out_idx = ((n * C + c) * H + h) * W + w;
    grad_input[out_idx] = gamma[c] * inv_std * (grad_output[out_idx] - x_hat * sum_dy_xhat / (H * W));
}

extern "C" void layer_norm_forward(float* d_output, float* d_input,
                                    float* d_gamma, float* d_beta,
                                    int N, int C, int H, int W, float eps) {
    int total = N * C * H * W;
    int tpb = 256;
    layer_norm_forward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_output, d_input, d_gamma, d_beta, N, C, H, W, eps);
    CUDA_KERNEL_CHECK();
}

extern "C" void layer_norm_backward(float* d_grad_input, float* d_grad_output,
                                      float* d_input, float* d_gamma,
                                      int N, int C, int H, int W, float eps) {
    int total = N * C * H * W;
    int tpb = 256;
    layer_norm_backward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_grad_input, d_grad_output, d_input, d_gamma, N, C, H, W, eps);
    CUDA_KERNEL_CHECK();
}
