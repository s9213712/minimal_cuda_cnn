# Minimal CUDA CNN

[English README](README.md)

Minimal CUDA CNN 是一個實驗性的 CUDA/C++ 與 Python 專案，用來訓練與除錯小型 CIFAR-10 CNN。主要實作路徑不依賴深度學習框架，而是透過手寫 CUDA kernel、C ABI shared library，以及 Python `ctypes` 呼叫完成訓練流程。

這個 repository 目前包含：

- 編譯成 `cpp/libminimal_cuda_cnn.so` 的 CUDA/C++ kernel
- Python `ctypes` shared object binding
- 手寫 CUDA CIFAR-10 訓練腳本
- 條件對齊的 PyTorch baseline
- `.so` 編譯、C API、Python/C++ 呼叫方式的教學文件

`bug.txt`、CIFAR-10 batch 檔、編譯後的 `.so`、`__pycache__` 與訓練 checkpoint 都是本機檔案，已透過 Git ignore 排除。

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
- 初始 learning rate：conv1、其他 conv、FC 都是 `0.002`
- Momentum：`0.9`
- Conv gradient spatial normalization：依照各層輸出 `H*W` 啟用

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

如果 CUDA 安裝路徑或 GPU 架構不同，請調整 `cpp/Makefile`：

```makefile
CC = /usr/local/cuda-13.2/bin/nvcc
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86
```

## 資料

請將 CIFAR-10 Python batch 放在：

```text
data/cifar-10-batches-py/
  data_batch_1
  data_batch_2
  data_batch_3
  data_batch_4
  data_batch_5
  test_batch
```

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

## 執行 PyTorch Baseline

```bash
cd /home/s92137/NN/minimal_cuda_cnn
python3 python/train_split_torch_baseline.py
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

目前 full-dataset 設定尚未完整 benchmark。此環境中 PyTorch 回報 `torch.cuda.is_available() == False`，所以上述 baseline 使用 CPU 跑完。若 PyTorch 能看到 CUDA device，腳本會自動改用 CUDA。

## Shared Object 文件

`.so` 使用教學拆成多份文件：

```text
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

檢查 `.so` 是否匯出 Momentum optimizer：

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep apply_momentum_update
```

## 備註

這仍是研究與除錯用 workspace。修改 CUDA kernel 時，建議先跑小型 kernel-level check，再跑 `python/train_split.py`，最後用 `python/train_split_torch_baseline.py` 做條件對齊比較。
