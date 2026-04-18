#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include "cuda_check.h"

// ============== Softmax CrossEntropy ==============
__global__ void softmax_kernel(const float* input, float* output, int N, int features) {
    int n = blockIdx.x;
    if (n >= N) return;
    
    const float* row = input + n * features;
    float* out_row = output + n * features;
    
    // Find max for numerical stability
    float max_val = -1e38f;
    for (int i = 0; i < features; i++) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    // Compute exp sum
    float sum = 0.0f;
    for (int i = 0; i < features; i++) {
        out_row[i] = expf(row[i] - max_val);
        sum += out_row[i];
    }
    
    // Normalize
    for (int i = 0; i < features; i++) {
        out_row[i] /= sum;
    }
}

__global__ void cross_entropy_backward_kernel(float* grad_out, float* probs, int N, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * features;
    if (idx >= total) return;
    
    // dL/dsoftmax = softmax - label (one-hot)
    // grad_out contains label (one-hot encoded)
    grad_out[idx] = (probs[idx] - grad_out[idx]);  // label[j=target] = 1, else 0
}

__global__ void softmax_xent_grad_loss_acc_kernel(
    const float* logits,
    const int* labels,
    float* probs,
    float* grad_logits,
    float* loss_sum,
    int* correct_count,
    int N,
    int features
) {
    int n = blockIdx.x;
    if (n >= N || threadIdx.x != 0) return;

    const float* row = logits + n * features;
    float* prob_row = probs + n * features;
    float* grad_row = grad_logits + n * features;
    int label = labels[n];

    float max_val = -1e38f;
    int pred = 0;
    for (int j = 0; j < features; ++j) {
        float v = row[j];
        if (v > max_val) {
            max_val = v;
            pred = j;
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < features; ++j) {
        float p = expf(row[j] - max_val);
        prob_row[j] = p;
        sum += p;
    }

    for (int j = 0; j < features; ++j) {
        float p = prob_row[j] / sum;
        prob_row[j] = p;
        float target = (j == label) ? 1.0f : 0.0f;
        grad_row[j] = (p - target) / static_cast<float>(N);
    }

    atomicAdd(loss_sum, -logf(prob_row[label] + 1e-10f));
    if (pred == label) {
        atomicAdd(correct_count, 1);
    }
}

__global__ void count_correct_kernel(
    const float* logits,
    const int* labels,
    int* correct_count,
    int N,
    int features
) {
    int n = blockIdx.x;
    if (n >= N || threadIdx.x != 0) return;

    const float* row = logits + n * features;
    int pred = 0;
    float best = row[0];
    for (int j = 1; j < features; ++j) {
        float v = row[j];
        if (v > best) {
            best = v;
            pred = j;
        }
    }
    if (pred == labels[n]) {
        atomicAdd(correct_count, 1);
    }
}

extern "C" {
    void softmax_forward(float* d_input, float* d_output, int N, int features) {
        int tpb = 256;
        softmax_kernel<<<N, tpb>>>(d_input, d_output, N, features);
        CUDA_KERNEL_CHECK();
    }
    
    float softmax_cross_entropy(float* d_input, float* d_labels, float* d_probs, float* d_loss, int N, int features) {
        // Forward: softmax
        softmax_forward(d_input, d_probs, N, features);

        // Copy to host for loss computation
        float* h_probs = (float*)malloc(N * features * sizeof(float));
        float* h_labels = (float*)malloc(N * features * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_probs, d_probs, N * features * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_labels, d_labels, N * features * sizeof(float), cudaMemcpyDeviceToHost));
        
        float loss = 0.0f;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < features; j++) {
                int idx = i * features + j;
                if (h_labels[idx] > 0.5f) {  // one-hot label
                    loss -= logf(h_probs[idx] + 1e-10f);
                }
            }
        }
        
        free(h_probs);
        free(h_labels);
        loss /= N;
        if (d_loss != nullptr) {
            CUDA_CHECK(cudaMemcpy(d_loss, &loss, sizeof(float), cudaMemcpyHostToDevice));
        }
        return loss;
    }
    
    void softmax_backward(float* d_grad_out, float* d_probs, int N, int features) {
        dim3 block(256);
        dim3 grid((N * features + 255) / 256);
        cross_entropy_backward_kernel<<<grid, block>>>(d_grad_out, d_probs, N, features);
        CUDA_KERNEL_CHECK();
    }

    void softmax_xent_grad_loss_acc(
        float* d_logits,
        int* d_labels,
        float* d_probs,
        float* d_grad_logits,
        float* d_loss_sum,
        int* d_correct_count,
        int N,
        int features
    ) {
        softmax_xent_grad_loss_acc_kernel<<<N, 1>>>(
            d_logits, d_labels, d_probs, d_grad_logits, d_loss_sum, d_correct_count, N, features
        );
        CUDA_KERNEL_CHECK();
    }

    void count_correct(float* d_logits, int* d_labels, int* d_correct_count, int N, int features) {
        count_correct_kernel<<<N, 1>>>(d_logits, d_labels, d_correct_count, N, features);
        CUDA_KERNEL_CHECK();
    }
}

// ============== Dense/FC Layer ==============
extern "C" {
    void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output, int N, int in_f, int out_f);
}

// ============== im2col Backward (Gradient to Input) ==============
__global__ void im2col_backward_kernel(float* grad_col, float* grad_input, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * KH * KW * N * outH * outW;
    if (idx >= total) return;
    
    int row = idx / (N * outH * outW);
    int col = idx % (N * outH * outW);
    
    int c = row / (KH * KW);
    int kh = (row / KW) % KH;
    int kw = row % KW;
    int n = col / (outH * outW);
    int ow = col % outW;
    int oh = (col / outW) % outH;
    
    int h_in = oh + kh;
    int w_in = ow + kw;
    
    // Accumulate gradient to input
    atomicAdd(&grad_input[((n * C + c) * H + h_in) * W + w_in], grad_col[idx]);
}

extern "C" {
    void im2col_backward(float* d_grad_col, float* d_grad_input, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
        int total = C * KH * KW * N * outH * outW;
        int tpb = 256;
        im2col_backward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_grad_col, d_grad_input, N, C, H, W, KH, KW, outH, outW);
        CUDA_KERNEL_CHECK();
    }
}

// ============== GEMM Backward (for FC gradient to weights) ==============
__global__ void gemm_backward_A_kernel(const float* grad_out, const float* B, float* grad_A, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += grad_out[row * N + i] * B[i * K + col];
        }
        grad_A[row * K + col] = sum;
    }
}

__global__ void gemm_backward_B_kernel(const float* A, const float* grad_out, float* grad_B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += A[i * K + row] * grad_out[i * N + col];
        }
        grad_B[row * N + col] = sum;
    }
}

extern "C" {
    void gemm_backward(float* d_grad_out, float* d_A, float* d_B, float* d_grad_A, float* d_grad_B, int M, int N, int K) {
        // grad_A = grad_out @ B^T  (M x K)
        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (M + 15) / 16);
        gemm_backward_A_kernel<<<grid, block>>>(d_grad_out, d_B, d_grad_A, M, N, K);
        
        // grad_B = A^T @ grad_out  (K x N)
        grid.x = (N + 15) / 16;
        grid.y = (K + 15) / 16;
        gemm_backward_B_kernel<<<grid, block>>>(d_A, d_grad_out, d_grad_B, M, N, K);
        
        CUDA_KERNEL_CHECK();
    }
}
