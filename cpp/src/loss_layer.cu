#include <cuda_runtime.h>
#include <cmath>

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
    
    int n = idx / features;
    int j = idx % features;
    
    // dL/dsoftmax = softmax - label (one-hot)
    // grad_out contains label (one-hot encoded)
    grad_out[idx] = (probs[idx] - grad_out[idx]);  // label[j=target] = 1, else 0
}

extern "C" {
    void softmax_forward(float* d_input, float* d_output, int N, int features) {
        int tpb = 256;
        softmax_kernel<<<N, tpb>>>(d_input, d_output, N, features);
        cudaDeviceSynchronize();
    }
    
    float softmax_cross_entropy(float* d_input, float* d_labels, float* d_probs, float* d_loss, int N, int features) {
        // Forward: softmax
        softmax_forward(d_input, d_probs, N, features);
        
        // Compute loss: -sum(label * log(probs))
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        cross_entropy_backward_kernel<<<grid, block>>>(d_labels, d_probs, N, features);
        cudaDeviceSynchronize();
        
        // Copy to host for loss computation
        float* h_probs = (float*)malloc(N * features * sizeof(float));
        float* h_labels = (float*)malloc(N * features * sizeof(float));
        cudaMemcpy(h_probs, d_probs, N * features * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_labels, d_labels, N * features * sizeof(float), cudaMemcpyDeviceToHost);
        
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
        return loss / N;
    }
    
    void softmax_backward(float* d_grad_out, float* d_probs, int N, int features) {
        dim3 block(256);
        dim3 grid((N * features + 255) / 256);
        cross_entropy_backward_kernel<<<grid, block>>>(d_grad_out, d_probs, N, features);
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
        
        cudaDeviceSynchronize();
    }
}
