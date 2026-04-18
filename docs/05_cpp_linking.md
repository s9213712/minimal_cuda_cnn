# C++ 連結與使用 `.so`

本文說明如何從 C++ 程式連結 `cpp/libminimal_cuda_cnn.so`。

## 基本方式

因為 `.so` 主要匯出 C ABI，C++ 程式可直接宣告相同 prototype：

```cpp
extern "C" {
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);

void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output,
                   int N, int in_f, int out_f);
}
```

## 最小 inference 範例

建立 `examples/mnist_infer_demo.cpp`：

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);

void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output,
                   int N, int in_f, int out_f);
}

int main() {
    constexpr int N = 2;
    constexpr int IN = 28 * 28;
    constexpr int OUT = 10;

    std::vector<float> x(N * IN, 0.0f);
    std::vector<float> w(OUT * IN, 0.001f);
    std::vector<float> b(OUT, 0.0f);
    std::vector<float> y(N * OUT, 0.0f);

    float* d_x = static_cast<float*>(gpu_malloc(x.size() * sizeof(float)));
    float* d_w = static_cast<float*>(gpu_malloc(w.size() * sizeof(float)));
    float* d_b = static_cast<float*>(gpu_malloc(b.size() * sizeof(float)));
    float* d_y = static_cast<float*>(gpu_malloc(y.size() * sizeof(float)));

    gpu_memcpy_h2d(d_x, x.data(), x.size() * sizeof(float));
    gpu_memcpy_h2d(d_w, w.data(), w.size() * sizeof(float));
    gpu_memcpy_h2d(d_b, b.data(), b.size() * sizeof(float));

    dense_forward(d_x, d_w, d_b, d_y, N, IN, OUT);
    gpu_memcpy_d2h(y.data(), d_y, y.size() * sizeof(float));

    for (int n = 0; n < N; ++n) {
        int argmax = 0;
        for (int j = 1; j < OUT; ++j) {
            if (y[n * OUT + j] > y[n * OUT + argmax]) argmax = j;
        }
        std::printf("sample %d pred=%d logit=%f\n", n, argmax, y[n * OUT + argmax]);
    }

    gpu_free(d_x);
    gpu_free(d_w);
    gpu_free(d_b);
    gpu_free(d_y);
    return 0;
}
```

編譯：

```bash
cd /home/s92137/NN/minimal_cuda_cnn
g++ examples/mnist_infer_demo.cpp \
  -Lcpp -lminimal_cuda_cnn \
  -Wl,-rpath,/home/s92137/NN/minimal_cuda_cnn/cpp \
  -o examples/mnist_infer_demo
```

執行：

```bash
./examples/mnist_infer_demo
```

## C++ 完整訓練流程

如果要從 C++ 做完整訓練，流程和 Python 相同：

1. 準備 NCHW float32 input，例如 MNIST 為 `(N, 1, 28, 28)`。
2. 用 `gpu_malloc` 配置 input、weights、bias、intermediate buffer。
3. Forward：`im2col_forward -> gemm_forward -> activation -> maxpool_forward_store -> layout convert -> dense_forward`。
4. 在 host 或 device 計算 loss gradient。若在 host 計算，將 logits copy 回 CPU，做 softmax/cross entropy，再 upload gradient。
5. Backward：`dense_backward_full -> layout convert -> maxpool_backward_use_idx -> activation backward -> conv_backward`。
6. Update：建議用 `apply_momentum_update` 更新 weights/bias；每個 trainable buffer 需要一個同長度 velocity buffer，訓練開始前清為 0 並跨 batch 保留。若只做最小測試，可用 `apply_sgd_update`。
7. 每個 batch 結束後釋放暫存 buffer。

Momentum update 的 C ABI prototype：

```cpp
extern "C" {
void apply_momentum_update(float* weights, float* grad, float* velocity,
                           float lr, float momentum, int size);
}
```

更新公式：

```text
velocity[i] = momentum * velocity[i] - lr * grad[i]
weights[i] += velocity[i]
```

## 注意事項

連結時最常見問題是 runtime 找不到 `.so`。有三種解法：

```bash
# 方式 1：編譯時寫 rpath
-Wl,-rpath,/home/s92137/NN/minimal_cuda_cnn/cpp

# 方式 2：執行前設定 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/s92137/NN/minimal_cuda_cnn/cpp:$LD_LIBRARY_PATH

# 方式 3：把 .so 放到系統 linker 搜尋路徑
```
