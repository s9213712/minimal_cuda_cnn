// MaxPool Forward with 2x2 stride - stores max indices
// Input: (C, N, H, W) — CNHW layout
// Output: (C, N, H/2, W/2) — CNHW layout  
// Stores max_idx in CNHW linear index space for backward pass

__global__ void maxpool_forward_store_kernel(const float* input, float* output, int* max_idx,
                                             int N, int C, int H, int W) {
    int out_h = H / 2;
    int out_w = W / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int n = (idx / (out_w * out_h)) % N;
    int c = idx / (out_w * out_h * N);

    int h_start = oh * 2;
    int w_start = ow * 2;

    float max_val = -1e38f;
    int max_linear_idx = -1;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int in_h = h_start + i;
            int in_w = w_start + j;
            // CNHW layout: (c, n, h, w) -> linear = ((c*N + n)*H + h)*W + w
            int linear_idx = ((c * N + n) * H + in_h) * W + in_w;
            if (input[linear_idx] > max_val) {
                max_val = input[linear_idx];
                max_linear_idx = linear_idx;
            }
        }
    }

    // Output is also in CNHW layout: (c, n, oh, ow) -> linear = ((c*N + n)*outH + oh)*outW + ow
    int out_idx = ((c * N + n) * out_h + oh) * out_w + ow;
    output[out_idx] = max_val;
    max_idx[out_idx] = max_linear_idx;
}

extern "C" void maxpool_forward_store(float* d_output, float* d_input, int* d_max_idx,
                                       int N, int C, int H, int W) {
    int out_h = H / 2;
    int out_w = W / 2;
    int total = N * C * out_h * out_w;
    int tpb = 256;
    // input and output are both CNHW layout
    maxpool_forward_store_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_input, d_output, d_max_idx, N, C, H, W);
    cudaDeviceSynchronize();
}
