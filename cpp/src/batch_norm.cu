#include "cuda_check.h"

// Batch Normalization Forward (training mode)
// input: (N, C, H, W)
// Computes batch mean, variance, normalizes, and stores running stats
// gamma, beta: learnable scale and shift (C channels)

__global__ void batch_norm_forward_kernel(float* output, const float* input,
                                         float* running_mean, float* running_var,
                                         float* saved_mean, float* saved_var,
                                         float* gamma, float* beta,
                                         float momentum, float eps,
                                         int N, int C, int H, int W, int train) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C;
    if (idx >= total) return;

    int c = idx;
    int spatial_size = N * H * W;

    // Compute batch mean over spatial
    float mean = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int in_idx = ((n * C + c) * H + h) * W + w;
                mean += input[in_idx];
            }
        }
    }
    mean /= spatial_size;

    // Compute batch variance
    float var = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int in_idx = ((n * C + c) * H + h) * W + w;
                float diff = input[in_idx] - mean;
                var += diff * diff;
            }
        }
    }
    var /= spatial_size;

    // Store for backward
    saved_mean[c] = mean;
    saved_var[c] = var;

    // Update running stats (for inference)
    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
    running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;

    // Normalize
    float inv_std = rsqrtf(var + eps);
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int in_idx = ((n * C + c) * H + h) * W + w;
                int out_idx = in_idx;
                output[out_idx] = gamma[c] * (input[in_idx] - mean) * inv_std + beta[c];
            }
        }
    }
}

__global__ void batch_norm_backward_kernel(float* grad_input, float* grad_output,
                                           const float* input, float* saved_mean, float* saved_var,
                                           float* gamma, float* grad_gamma, float* grad_beta,
                                           float eps, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float mean = saved_mean[c];
    float var = saved_var[c];
    float inv_std = rsqrtf(var + eps);
    float x_norm = (input[idx] - mean) * inv_std;
    float dy = grad_output[idx];

    // Grad to gamma and beta (accumulate across spatial)
    atomicAdd(&grad_gamma[c], dy * x_norm);
    atomicAdd(&grad_beta[c], dy);

    // dL/dx
    float spatial = (float)(H * W);
    float g_norm = dy * gamma[c] * inv_std;
    
    // Subtract mean from grad
    float sum_dy_centered = 0.0f;
    for (int h2 = 0; h2 < H; h2++) {
        for (int w2 = 0; w2 < W; w2++) {
            int idx2 = ((n * C + c) * H + h2) * W + w2;
            sum_dy_centered += grad_output[idx2];
        }
    }
    
    float grad_var = 0.0f;
    for (int h2 = 0; h2 < H; h2++) {
        for (int w2 = 0; w2 < W; w2++) {
            int idx2 = ((n * C + c) * H + h2) * W + w2;
            float centered = (input[idx2] - mean) * inv_std;
            grad_var += grad_output[idx2] * gamma[c] * centered * (-0.5f) * powf(var + eps, -1.5f);
        }
    }
    
    float grad_mean = sum_dy_centered * gamma[c] * inv_std + grad_var * (-2.0f / spatial) * (input[idx] - mean);
    
    grad_input[idx] = g_norm - grad_mean / spatial - (input[idx] - mean) * 2.0f * grad_var / spatial;
}

extern "C" {
    void batch_norm_forward(float* d_output, float* d_input,
                           float* d_running_mean, float* d_running_var,
                           float* d_saved_mean, float* d_saved_var,
                           float* d_gamma, float* d_beta,
                           float momentum, float eps, int N, int C, int H, int W, int train) {
        int tpb = 256;
        batch_norm_forward_kernel<<<C, tpb>>>(d_output, d_input, d_running_mean, d_running_var,
                                              d_saved_mean, d_saved_var, d_gamma, d_beta,
                                              momentum, eps, N, C, H, W, train);
        CUDA_KERNEL_CHECK();
    }

    void batch_norm_backward(float* d_grad_input, float* d_grad_output,
                             const float* d_input, float* d_saved_mean, float* d_saved_var,
                             float* d_gamma, float* d_grad_gamma, float* d_grad_beta,
                             float eps, int N, int C, int H, int W) {
        int total = C * N * H * W;
        int tpb = 256;
        // Init grad_gamma and grad_beta to 0
        CUDA_CHECK(cudaMemset(d_grad_gamma, 0, C * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_beta, 0, C * sizeof(float)));
        batch_norm_backward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_grad_input, d_grad_output,
                                                                      d_input, d_saved_mean, d_saved_var,
                                                                      d_gamma, d_grad_gamma, d_grad_beta,
                                                                      eps, N, C, H, W);
        CUDA_KERNEL_CHECK();
    }
}
