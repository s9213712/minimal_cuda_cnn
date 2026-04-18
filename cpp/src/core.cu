#include "tensor.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#ifndef USE_CUBLAS
#define USE_CUBLAS 1
#endif

#if USE_CUBLAS
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#endif
#include <iostream>
#include <vector>

#if USE_CUBLAS
static cublasHandle_t g_cublas = nullptr;

static void cublas_check(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: %s failed with status %d\n",
                     file, line, expr, static_cast<int>(status));
        std::fflush(stderr);
        std::abort();
    }
}

#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __FILE__, __LINE__)

static cublasHandle_t get_cublas() {
    if (!g_cublas) {
        CUBLAS_CHECK(cublasCreate(&g_cublas));
    }
    return g_cublas;
}
#endif

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
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C_out[row * N + col] = sum;
    }
}

extern "C" {
    void im2col_forward(float* d_input, float* d_output, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
        int total_elements = (C * KH * KW) * (N * outH * outW);
        int tpb = 256;
        im2col_kernel<<<(total_elements + tpb - 1) / tpb, tpb>>>(d_input, d_output, N, C, H, W, KH, KW, outH, outW);
        CUDA_KERNEL_CHECK();
    }

    void gemm_forward(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
#if USE_CUBLAS
        cublasHandle_t handle = get_cublas();
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Row-major C[M, N] = A[M, K] * B[K, N].
        // cuBLAS is column-major, so compute C^T[N, M] = B^T[N, K] * A^T[K, M].
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            d_A,
            K,
            &beta,
            d_C,
            N
        ));
#else
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_KERNEL_CHECK();
#endif
    }

    void apply_relu(float* d_data, int size) {
        int tpb = 256;
        relu_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        CUDA_KERNEL_CHECK();
    }

    void apply_maxpool(float* d_input, float* d_output, int n, int c, int h, int w) {
        int out_h = h / 2; int out_w = w / 2;
        int size = n * c * out_h * out_w;
        int tpb = 256;
        maxpool_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_input, d_output, n, c, h, w);
        CUDA_KERNEL_CHECK();
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
