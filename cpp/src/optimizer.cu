__global__ void sgd_update_kernel(float* weights, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

extern "C" {
    void apply_sgd_update(float* d_weights, float* d_grad, float lr, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        sgd_update_kernel<<<bpg, tpb>>>(d_weights, d_grad, lr, size);
        cudaDeviceSynchronize();
    }
}
