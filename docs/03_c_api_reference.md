# C API 參考

`libminimal_cuda_cnn.so` 主要透過 `extern "C"` 匯出函式，適合 Python `ctypes`、C/C++ 或其他 FFI 呼叫。

## GPU 記憶體

```c
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);
void gpu_memset(void* dst, int value, size_t size);
```

`size` 是 bytes，不是元素數。`float32` buffer 大小通常是 `num_elements * 4`。

## Forward API

```c
void im2col_forward(float* input, float* col,
                    int N, int C, int H, int W,
                    int KH, int KW, int outH, int outW);

void gemm_forward(float* A, float* B, float* C,
                  int M, int N, int K);

void dense_forward(float* input, float* weights, float* bias, float* output,
                   int N, int in_f, int out_f);

void apply_relu(float* data, int size);
void leaky_relu_forward(float* data, float alpha, int size);

void maxpool_forward_store(float* output, float* input, int* max_idx,
                           int N, int C, int H, int W);

void softmax_forward(float* input, float* output, int N, int features);
```

典型 convolution forward：

```text
NCHW input
-> im2col_forward
-> gemm_forward(weights, col, conv_raw)
-> activation
```

`gemm_forward` 以 row-major API 暴露。`USE_CUBLAS=1` 時內部使用 cuBLAS `cublasSgemm`；`USE_CUBLAS=0` 時使用手寫 CUDA GEMM kernel。Conv output 常用作 CNHW：`(OUT_C, N, outH, outW)`。

## Backward API

```c
void dense_backward_full(float* d_out, float* input, float* weights,
                         float* d_input, float* d_weights, float* d_bias,
                         int N, int in_f, int out_f);

void conv_backward(float* grad_out, float* input, float* weights,
                   float* grad_weights, float* grad_input,
                   int N, int C, int H, int W,
                   int KH, int KW, int outH, int outW, int OUT_C);

void conv_backward_precol(float* grad_out, float* input, float* weights,
                          float* grad_weights, float* grad_input,
                          float* col,
                          int N, int C, int H, int W,
                          int KH, int KW, int outH, int outW, int OUT_C);

void leaky_relu_backward(float* data, float* grad, float alpha, int size);

void maxpool_backward_use_idx(float* grad_out, int* max_idx, float* grad_input,
                              int N, int C, int H, int W);

void softmax_backward(float* grad_out, float* probs, int N, int features);
```

`dense_backward_full` 預期 `d_out` 已經包含 loss reduction 的縮放。例如 softmax cross entropy 若使用 batch mean，傳入前應先做 `(probs - labels) / N`。函式本身不再額外除以 `N`，因此可直接搭配 Python/NumPy 或 GPU 端已平均過的 logits gradient。

`conv_backward` 會計算：

```text
grad_weights: dL/dW
grad_input:   dL/dInput
```

`USE_CUBLAS=1` 時，`grad_weights` 使用 im2col + cuBLAS GEMM 計算，避免舊版 per-element `atomicAdd` 累積造成的嚴重 contention。`USE_CUBLAS=0` 時會使用手寫 CUDA fallback，不需要連結 cuBLAS。`grad_input` 仍使用直接 CUDA kernel。訓練程式會對部分 gradient 做 clipping 或 batch 平均，這些縮放通常在 Python 端或 optimizer kernel 完成。

`conv_backward_precol` 與 `conv_backward` 行為相同，但多接收一個已存在的 `col` buffer。當 forward 已經呼叫過 `im2col_forward(input, col, ...)` 時，訓練 loop 可直接把同一個 `col` 傳入 backward，避免重新 im2col 與額外配置 scratch buffer。

## Optimizer

```c
void apply_sgd_update(float* weights, float* grad, float lr, int size);

void apply_momentum_update(float* weights, float* grad, float* velocity,
                           float lr, float momentum, int size);

void conv_update_fused(float* weights, float* grad, float* velocity,
                       float lr, float momentum, float weight_decay,
                       float clip_val, float normalizer, int size);

void clip_inplace(float* values, float clip_val, int size);
```

`apply_sgd_update` 行為：

```text
weights[i] -= lr * grad[i]
```

`apply_momentum_update` 行為：

```text
velocity[i] = momentum * velocity[i] - lr * grad[i]
weights[i] += velocity[i]
```

`velocity` 必須是和 `weights` 相同長度的 GPU buffer，訓練開始前初始化為 0，並在整個訓練期間保留。

`conv_update_fused` 會在 GPU 端完成：

```text
g = grad[i] / normalizer + weight_decay * weights[i]
g = clip(g, -clip_val, clip_val)
velocity[i] = momentum * velocity[i] - lr * g
weights[i] += velocity[i]
```

雖然函式名稱保留 `conv`，目前也可用於 FC weight/bias update；bias update 時將 `weight_decay` 設成 `0.0`。

`clip_inplace` 會直接在 GPU buffer 上做 in-place clipping，可用於 activation/feature gradient。

## Layout 轉換

```c
void nchw_to_cnhw(float* input, float* output, int N, int C, int H, int W);
void cnhw_to_nchw(float* input, float* output, int N, int C, int H, int W);
```

使用重點：

```text
im2col_forward input: NCHW
conv raw output: CNHW
maxpool_forward_store input/output: CNHW
dense_forward input: row-major (N, features)
```

## Softmax loss

```c
float softmax_cross_entropy(float* input, float* labels, float* probs, float* loss,
                            int N, int features);
```

`labels` 是 one-hot label。此函式會計算 softmax 與 loss，不會修改 label buffer。若 `loss` 非 null，會把 scalar loss 寫到 device pointer。
