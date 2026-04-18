# Minimal CUDA CNN

[English README](README.md)

Minimal CUDA CNN 是一個實驗性的 CUDA/C++ 與 Python 專案，用來訓練與除錯小型 CIFAR-10 CNN。主要實作路徑不依賴深度學習框架，而是透過手寫 CUDA kernel、C ABI shared library，以及 Python `ctypes` 呼叫完成訓練流程。

這個 repository 目前包含：

- 編譯成 `cpp/libminimal_cuda_cnn.so` 的 CUDA/C++ kernel
- Python `ctypes` shared object binding
- 手寫 CUDA CIFAR-10 訓練腳本
- 條件對齊的 PyTorch baseline
- `.so` 編譯、C API、Python/C++ 呼叫方式的教學文件

CIFAR-10 batch 檔、編譯後的 `.so`、`__pycache__` 與訓練 checkpoint 都是本機檔案，已透過 Git ignore 排除。

## 目前 CIFAR-10 實驗

手寫 CUDA trainer 和 PyTorch baseline 使用相同比較條件：

- Dataset：`data/cifar-10-batches-py` 內的 CIFAR-10 Python batch
- 訓練來源：`data_batch_1` 到 `data_batch_5`
- 切分：`45000` train / `5000` validation
- Test：官方 `test_batch`
- Dataset split seed：`42`
- Initial weight seed：`42`
- Input normalization：CIFAR-10 channel mean/std
- Batch size：`64`
- Max epochs：`50`
- Early stopping patience：`8`
- 架構：valid 3x3 convolution stack，不使用 padding
- Activation：LeakyReLU，alpha `0.1`
- Optimizer：手動 Momentum SGD update、weight decay、gradient clipping、LR plateau reduction
- 初始 learning rate：conv1、其他 conv、FC 都是 `0.005`
- Momentum：`0.9`
- Conv gradient spatial normalization：目前關閉

模型架構：

```text
Conv(3->32, 3x3 valid) -> LeakyReLU
Conv(32->32, 3x3 valid) -> LeakyReLU
MaxPool(2x2)
Conv(32->64, 3x3 valid) -> LeakyReLU
Conv(64->64, 3x3 valid) -> LeakyReLU
MaxPool(2x2)
FC(1600->10)
```

Feature map shape：

```text
32x32 -> 30x30 -> 28x28 -> 14x14 -> 12x12 -> 10x10 -> 5x5
```

## Python 檔案結構

```text
python/
  train_split.py                 手寫 CUDA CIFAR-10 trainer
  train_split_torch_baseline.py  條件對齊的 PyTorch baseline
  train_config.py                共用訓練參數與模型尺寸
  cifar10_data.py                CIFAR-10 讀取、split、normalization
  model_init.py                  共用 CPU 端初始權重
  cuda_backend.py                ctypes binding 與 GPU helper
  model_forward.py               CUDA inference/evaluation helper
  model_weights.py               CUDA weight upload/checkpoint/cleanup helper
```

`train_split.py` 主要保留訓練 loop 與 backward pass。資料、參數、初始化、forward/evaluate、CUDA binding 都已拆成獨立檔案，方便和 PyTorch baseline 使用相同條件比較。

## 環境需求

- NVIDIA GPU，供手寫 CUDA trainer 使用
- CUDA toolkit
- 可選 cuBLAS，通常隨 CUDA toolkit 安裝。預設啟用以加速，但保留純手寫 CUDA fallback。
- Python 3
- NumPy
- PyTorch，僅供 `python/train_split_torch_baseline.py` baseline 使用

CUDA 實作本身不依賴 PyTorch。PyTorch 只作為 baseline/reference trainer。

## 編譯

編譯 CUDA shared library：

```bash
cd /home/s92137/NN/minimal_cuda_cnn
make -C cpp
```

成功後會產生：

```text
cpp/libminimal_cuda_cnn.so
```

預設 `.so` 會連結 cuBLAS，`gemm_forward` 與 `conv_backward` 的 conv weight gradient 路徑都使用 `cublasSgemm`。

編譯快速 cuBLAS backend：

```bash
make -C cpp USE_CUBLAS=1
```

編譯不連結 cuBLAS 的 from-scratch fallback：

```bash
make -C cpp USE_CUBLAS=0
```

如果 CUDA 安裝路徑或 GPU 架構不同，請調整 `cpp/Makefile`：

```makefile
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86
USE_CUBLAS ?= 1
```

檢查 `.so` 是否建置成功且匯出必要 API：

```bash
make -C cpp check
```

## 資料

trainer 可自動準備 CIFAR-10：

```bash
python3 python/prepare_cifar10.py
```

此命令會下載並解壓 CIFAR-10 Python archive 到：

```text
data/cifar-10-batches-py/
  data_batch_1
  data_batch_2
  data_batch_3
  data_batch_4
  data_batch_5
  test_batch
```

如果機器不能連網，請手動將 CIFAR-10 Python batch 解壓後放在同一個目錄。

目前實驗使用全部五個 CIFAR-10 training batch，設定在 `python/train_config.py`：

```python
N_TRAIN = 45000
N_VAL = 5000
TRAIN_BATCH_IDS = (1, 2, 3, 4, 5)
```

## 執行 CUDA Trainer

```bash
cd /home/s92137/NN/minimal_cuda_cnn
python3 python/train_split.py
```

CUDA trainer 會將最佳本機 checkpoint 存到：

```text
python/best_model_split.npz
```

這個 checkpoint 已被 Git ignore。

目前 `train_split.py` 已把主要 FC backward 與 optimizer update 留在 GPU：

- `dense_forward` 直接使用 `d_pool2_nchw`，不再做 `d_pool2_nchw -> CPU -> GPU`
- `dense_backward_full` 產生 FC weight/bias gradient 與 pool gradient
- FC/Conv 參數更新使用 GPU fused momentum update、weight decay、gradient clipping
- pool gradient clipping 使用 GPU in-place clipping
- softmax、cross-entropy gradient、loss 累加與 batch accuracy 都在 CUDA 端執行；Python 只下載 loss/correct scalar
- conv forward GEMM 在 `USE_CUBLAS=1` 時使用 cuBLAS，在 `USE_CUBLAS=0` 時使用手寫 GEMM kernel
- conv backward 的 weight gradient 在 `USE_CUBLAS=1` 時使用 im2col + cuBLAS；`USE_CUBLAS=0` 時仍保留手寫 fallback
- 每個 batch 的固定 shape GPU buffer 由 `BatchWorkspace` 預先配置並重用，不再在 batch loop 內反覆配置/釋放
- train/eval 支援 remainder batch，不再丟掉最後不足 `BATCH` 的資料
- conv backward 會重用 forward 已產生的 im2col buffer，避免 backward 重新 im2col 與額外配置

2026-04-18 `USE_CUBLAS=1` smoke test：`timeout 70s python3 -u python/train_split.py` 可跑到第 6 epoch，單 epoch約 `8.6-14.8s`，第 6 epoch validation accuracy 為 `73.14%`。這不是完整 benchmark，但足以確認速度已從原本 `Batch 100/703` 約兩分鐘降到數秒內。

## 執行 PyTorch Baseline

```bash
cd /home/s92137/NN/minimal_cuda_cnn
python3 python/train_split_torch_baseline.py
```

如果目前環境的 CUDA/NVML 初始化會導致 PyTorch crash 或 segmentation fault，可先強制使用 CPU：

```bash
FORCE_CPU=1 python3 python/train_split_torch_baseline.py
```

PyTorch baseline 會將最佳本機 checkpoint 存到：

```text
python/best_model_split_torch.pt
```

這個 checkpoint 已被 Git ignore。

2026-04-18 舊版 `data_batch_1`-only Momentum baseline 結果：

```text
Best validation accuracy: 63.26% at epoch 25
Official test accuracy:   62.16%
Early stopping:           epoch 33
Device used:              CPU
```

目前 full-dataset 設定尚未完整 benchmark。此環境中 PyTorch 回報 `torch.cuda.is_available() == False` 且 NVML 初始化失敗；若 PyTorch 能看到 CUDA device，腳本會自動改用 CUDA。

## Shared Object 文件

`.so` 使用教學拆成多份文件：

```text
docs/USAGE.md
docs/01_project_files.md
docs/02_build_shared_library.md
docs/03_c_api_reference.md
docs/04_python_ctypes_mnist.md
docs/05_cpp_linking.md
docs/06_layout_and_debug.md
docs/train_mnist_so.py
```

這些文件說明 `.cu`、`.h` 檔案用途，如何編譯 `.so`，以及如何從 Python 或 C++ 呼叫。

## 常用檢查

語法檢查：

```bash
python3 -m py_compile python/*.py docs/train_mnist_so.py
```

檢查 PyTorch baseline 與共用初始化權重是否完全一致：

```bash
python3 -c "import sys, torch; sys.path.insert(0, 'python'); from model_init import init_weights; from train_split_torch_baseline import TorchCifarCnn, load_initial_weights; from train_config import INIT_SEED; m=TorchCifarCnn(); load_initial_weights(m, torch.device('cpu')); w=init_weights(INIT_SEED); print((m.conv1.weight.detach().flatten()-torch.from_numpy(w[0])).abs().max().item())"
```

預期輸出：

```text
0.0
```

檢查 `.so` 是否匯出 Momentum optimizer 與 fused update：

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep -E 'apply_momentum_update|conv_update_fused|clip_inplace'
```

## 備註

這仍是研究與除錯用 workspace。修改 CUDA kernel 時，建議先跑小型 kernel-level check，再跑 `python/train_split.py`，最後用 `python/train_split_torch_baseline.py` 做條件對齊比較。
