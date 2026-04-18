CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
USE_CUBLAS ?= 1
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86 -DUSE_CUBLAS=$(USE_CUBLAS)
LDFLAGS = -shared -o libminimal_cuda_cnn.so -Xlinker -rpath,$(CUDA_HOME)/lib64
EXTRA_LDFLAGS ?=

ifeq ($(USE_CUBLAS),1)
CUBLAS_LDFLAGS = -lcublas
else
CUBLAS_LDFLAGS =
endif

SRCS = \
	src/core.cu \
	src/gpu_monitor.cu \
	src/network.cu \
	src/dense_layer.cu \
	src/backward.cu \
	src/memory.cu \
	src/loss_layer.cu \
	src/conv_backward.cu \
	src/optimizer.cu \
	src/reorganize.cu \
	src/reorganize_backward.cu \
	src/maxpool_backward_nchw.cu \
	src/leaky_relu.cu \
	src/layer_norm.cu \
	src/maxpool_store.cu \
	src/maxpool_backward_use_idx.cu \
	src/layout_convert.cu
INCS = -Iinclude

.PHONY: all clean check

all:
	$(NVCC) $(CFLAGS) $(INCS) $(SRCS) $(LDFLAGS) $(CUBLAS_LDFLAGS) $(EXTRA_LDFLAGS)

check: all
	@test -f libminimal_cuda_cnn.so
	@nm -D libminimal_cuda_cnn.so | grep -q ' gpu_malloc'
	@nm -D libminimal_cuda_cnn.so | grep -q ' gemm_forward'
	@nm -D libminimal_cuda_cnn.so | grep -q ' conv_backward_precol'
	@nm -D libminimal_cuda_cnn.so | grep -q ' softmax_xent_grad_loss_acc'
	@echo "libminimal_cuda_cnn.so build check passed (USE_CUBLAS=$(USE_CUBLAS))"

clean:
	rm -f *.so src/*.o
