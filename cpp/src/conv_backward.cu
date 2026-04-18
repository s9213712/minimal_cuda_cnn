#include "cuda_check.h"
#include <cuda_runtime.h>
#ifndef USE_CUBLAS
#define USE_CUBLAS 1
#endif

#if USE_CUBLAS
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

static cublasHandle_t g_conv_cublas = nullptr;

static void cublas_check(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: %s failed with status %d\n",
                     file, line, expr, static_cast<int>(status));
        std::fflush(stderr);
        std::abort();
    }
}

#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __FILE__, __LINE__)

static cublasHandle_t get_conv_cublas() {
    if (!g_conv_cublas) {
        CUBLAS_CHECK(cublasCreate(&g_conv_cublas));
    }
    return g_conv_cublas;
}
#endif

static void conv_backward_with_col(
    float* grad_out,
    float* col,
    float* input,
    float* weights,
    float* grad_weights,
    float* grad_input,
    int N,
    int C,
    int H,
    int W,
    int KH,
    int KW,
    int outH,
    int outW,
    int OUT_C
);

__global__ void conv_backward_weight_atomic_kernel(float* grad_out, float* input, float* grad_weights,
                                                    int N, int C, int H, int W, int KH, int KW,
                                                    int outH, int outW, int OUT_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = OUT_C * C * KH * KW * N * outH * outW;
    if (idx >= total_elements) return;

    int row = idx / (N * outH * outW);
    int col_idx = idx % (N * outH * outW);

    int oc = row / (C * KH * KW);
    int c = (row / (KH * KW)) % C;
    int kh = (row / KW) % KH;
    int kw = row % KW;

    int n = col_idx / (outH * outW);
    int ow = col_idx % outW;
    int oh = (col_idx / outW) % outH;

    int w_idx = oc * (C * KH * KW) + c * (KH * KW) + kh * KW + kw;
    int x_idx = ((n * C + c) * H + (oh + kh)) * W + (ow + kw);
    int g_idx = oc * (N * outH * outW) + col_idx;
    atomicAdd(&grad_weights[w_idx], input[x_idx] * grad_out[g_idx]);
}

__global__ void conv_backward_weight_precol_kernel(float* grad_out, float* col, float* grad_weights,
                                                    int patch_size, int spatial_size, int OUT_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = OUT_C * patch_size;
    if (idx >= total_elements) return;

    int oc = idx / patch_size;
    int k = idx % patch_size;
    float sum = 0.0f;
    for (int p = 0; p < spatial_size; ++p) {
        sum += grad_out[oc * spatial_size + p] * col[k * spatial_size + p];
    }
    grad_weights[idx] = sum;
}

__global__ void conv_backward_im2col_kernel(const float* input, float* col,
                                             int N, int C, int H, int W,
                                             int KH, int KW, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = C * KH * KW * N * outH * outW;
    if (idx >= total_elements) return;

    int row = idx / (N * outH * outW);
    int col_idx = idx % (N * outH * outW);
    int c = (row / (KH * KW)) % C;
    int kh = (row / KW) % KH;
    int kw = row % KW;

    int n = col_idx / (outH * outW);
    int ow = col_idx % outW;
    int oh = (col_idx / outW) % outH;

    col[idx] = input[((n * C + c) * H + (oh + kh)) * W + (ow + kw)];
}

// Input gradient: d_input[n,c,h,w] += sum over oc,kh,kw of: d_conv2_grad[oc,n,oh,ow] * weights[oc,c,kh,kw]
// where oh = h - kh, ow = w - kw (only if oh,ow are valid)
__global__ void conv_backward_input_kernel(float* grad_out, float* weights, float* grad_input,
                                            int N, int C, int H, int W, int KH, int KW, int outH, int outW, int OUT_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;  // one thread per input element
    if (idx >= total_elements) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float sum = 0.0f;
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int kh = 0; kh < KH; kh++) {
            int oh = h - kh;
            if (oh < 0 || oh >= outH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int ow = w - kw;
                if (ow < 0 || ow >= outW) continue;
                // grad_out[oc, n, oh, ow]
                // weights[oc, c, kh, kw]
                int g_out_idx = oc * (N * outH * outW) + n * (outH * outW) + oh * outW + ow;
                int w_idx = oc * (C * KH * KW) + c * (KH * KW) + kh * KW + kw;
                sum += grad_out[g_out_idx] * weights[w_idx];
            }
        }
    }
    grad_input[idx] = sum;
}

static void conv_backward_with_col(
    float* grad_out,
    float* col,
    float* input,
    float* weights,
    float* grad_weights,
    float* grad_input,
    int N,
    int C,
    int H,
    int W,
    int KH,
    int KW,
    int outH,
    int outW,
    int OUT_C
) {
    int patch_size = C * KH * KW;
    int spatial_size = N * outH * outW;
    int tpb = 256;

#if USE_CUBLAS
    cublasHandle_t handle = get_conv_cublas();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        patch_size,
        OUT_C,
        spatial_size,
        &alpha,
        col,
        spatial_size,
        grad_out,
        spatial_size,
        &beta,
        grad_weights,
        patch_size
    ));
#else
    conv_backward_weight_precol_kernel<<<(OUT_C * patch_size + tpb - 1) / tpb, tpb>>>(
        grad_out, col, grad_weights, patch_size, spatial_size, OUT_C
    );
    CUDA_CHECK(cudaGetLastError());
#endif

    int total_in = N * C * H * W;
    conv_backward_input_kernel<<<(total_in + tpb - 1) / tpb, tpb>>>(
        grad_out, weights, grad_input, N, C, H, W, KH, KW, outH, outW, OUT_C
    );
    CUDA_KERNEL_CHECK();
}

extern "C" {
    void conv_backward_precol(float* grad_out, float* input, float* weights, float* grad_weights, float* grad_input,
                              float* col, int N, int C, int H, int W, int KH, int KW, int outH, int outW, int OUT_C) {
        conv_backward_with_col(
            grad_out, col, input, weights, grad_weights, grad_input,
            N, C, H, W, KH, KW, outH, outW, OUT_C
        );
    }

    void conv_backward(float* grad_out, float* input, float* weights, float* grad_weights, float* grad_input,
                       int N, int C, int H, int W, int KH, int KW, int outH, int outW, int OUT_C) {
#if USE_CUBLAS
        int patch_size = C * KH * KW;
        int spatial_size = N * outH * outW;
        int tpb = 256;
        float* col = nullptr;
        CUDA_CHECK(cudaMalloc(&col, patch_size * spatial_size * sizeof(float)));
        conv_backward_im2col_kernel<<<(patch_size * spatial_size + tpb - 1) / tpb, tpb>>>(
            input, col, N, C, H, W, KH, KW, outH, outW
        );
        CUDA_CHECK(cudaGetLastError());

        conv_backward_with_col(
            grad_out, col, input, weights, grad_weights, grad_input,
            N, C, H, W, KH, KW, outH, outW, OUT_C
        );
        CUDA_CHECK(cudaFree(col));
#else
        int tpb = 256;
        int total_w = OUT_C * C * KH * KW * N * outH * outW;
        CUDA_CHECK(cudaMemset(grad_weights, 0, OUT_C * C * KH * KW * sizeof(float)));
        conv_backward_weight_atomic_kernel<<<(total_w + tpb - 1) / tpb, tpb>>>(
            grad_out, input, grad_weights, N, C, H, W, KH, KW, outH, outW, OUT_C
        );
        CUDA_CHECK(cudaGetLastError());

        int total_in = N * C * H * W;
        conv_backward_input_kernel<<<(total_in + tpb - 1) / tpb, tpb>>>(
            grad_out, weights, grad_input, N, C, H, W, KH, KW, outH, outW, OUT_C
        );
        CUDA_KERNEL_CHECK();
#endif
    }
}
