NVCC ?= nvcc
CUDA_ARCH ?= 86
CUDA_PTX ?= $(CUDA_ARCH)
NVCCFLAGS ?= -O3 -std=c++14 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_PTX),code=compute_$(CUDA_PTX)

TARGET = quasi_cuda
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
