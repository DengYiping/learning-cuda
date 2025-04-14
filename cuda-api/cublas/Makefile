# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -arch=sm_90 -O3

# Linker flags
LDFLAGS := -lcublas -lcublasLt

# Source directory
SRC_DIR := .

# Executables
TARGETS := gemm gemm_cublaslt gemm_cublasxt

all: $(TARGETS)

gemm: $(SRC_DIR)/gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LDFLAGS)

gemm_cublaslt: $(SRC_DIR)/gemm_cublaslt.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LDFLAGS)

gemm_cublasxt: $(SRC_DIR)/gemm_cublasxt.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean