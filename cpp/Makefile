CC = /usr/local/cuda-13.2/bin/nvcc
CXX = g++
USE_CUBLAS ?= 1
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86 -DUSE_CUBLAS=$(USE_CUBLAS)
LDFLAGS = -shared -o libminimal_cuda_cnn.so -Xlinker -rpath,/usr/local/cuda-13.2/lib64
EXTRA_LDFLAGS ?=

ifeq ($(USE_CUBLAS),1)
CUBLAS_LDFLAGS = -lcublas
else
CUBLAS_LDFLAGS =
endif

SRCS = src/core.cu src/gpu_monitor.cu src/network.cu src/dense_layer.cu src/backward.cu src/memory.cu src/loss_layer.cu src/reorganize.cu src/reorganize_backward.cu src/maxpool_backward_nchw.cu src/leaky_relu.cu src/layer_norm.cu src/maxpool_store.cu src/maxpool_backward_use_idx.cu src/layout_convert.cu src/conv_backward.cu src/optimizer.cu
INCS = -Iinclude

all:
	$(CC) $(CFLAGS) $(INCS) src/core.cu src/gpu_monitor.cu src/network.cu src/dense_layer.cu src/backward.cu src/memory.cu src/loss_layer.cu src/conv_backward.cu src/optimizer.cu src/reorganize.cu src/reorganize_backward.cu src/maxpool_backward_nchw.cu src/leaky_relu.cu src/layer_norm.cu src/maxpool_store.cu src/maxpool_backward_use_idx.cu src/layout_convert.cu $(LDFLAGS) $(CUBLAS_LDFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f *.so src/*.o
