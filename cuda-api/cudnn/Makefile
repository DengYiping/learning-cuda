NVCC = nvcc
CFLAGS = -O3 -std=c++14
ARCH = -arch=sm_90
CUDNN_FLAGS = -lcudnn

all: tanh conv2d

tanh: tanh.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDNN_FLAGS) -o tanh tanh.cu

conv2d: cov2d.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDNN_FLAGS) -o conv2d cov2d.cu

clean:
	rm -f tanh conv2d

.PHONY: all clean