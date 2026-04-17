// MaxPool Backward using stored max indices from forward pass
// grad_out: (C, N, H/2, W/2) — CNHW layout (same as forward output)
// grad_input: (C, N, H, W) — CNHW layout, initialized to zeros before calling
// max_idx: (C, N, H/2, W/2) — CNHW linear indices into grad_input space

__global__ void maxpool_backward_use_idx_kernel(const float* grad_out, const int* max_idx,
                                                float* grad_input, int N, int C, int H, int W) {
    int poolH = H / 2;
    int poolW = W / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * poolH * poolW;
    if (idx >= total) return;

    // grad_out and max_idx are both in CNHW layout (same as forward output layout)
    // max_idx stores CNHW linear indices pointing into grad_input
    int linear_idx = max_idx[idx];
    atomicAdd(&grad_input[linear_idx], grad_out[idx]);
}

extern "C" void maxpool_backward_use_idx(float* d_grad_out, int* d_max_idx,
                                          float* d_grad_input, int N, int C, int H, int W) {
    int poolH = H / 2;
    int poolW = W / 2;
    int total = N * C * poolH * poolW;
    int tpb = 256;
    maxpool_backward_use_idx_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_grad_out, d_max_idx, d_grad_input, N, C, H, W);
    cudaDeviceSynchronize();
}
