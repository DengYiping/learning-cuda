NVCC = nvcc
NVCCFLAGS = -O3 -std=c++20 -arch=sm_90 -Xcompiler -Wall,-Wextra
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra

all: naive_matmul_benchmark coalesced_matmul_benchmark sram_matmul_benchmark sram_bank_conflict_free_matmul_benchmark

naive_matmul_benchmark: naive_matmul.cu
	$(NVCC) $(NVCCFLAGS) naive_matmul.cu -o naive_matmul_benchmark

coalesced_matmul_benchmark: coalesced_matmul.cu
	$(NVCC) $(NVCCFLAGS) coalesced_matmul.cu -o coalesced_matmul_benchmark

sram_matmul_benchmark: sram_matmul.cu
	$(NVCC) $(NVCCFLAGS) sram_matmul.cu -o sram_matmul_benchmark

sram_bank_conflict_free_matmul_benchmark: sram_bank_conflict_free_matmul.cu
	$(NVCC) $(NVCCFLAGS) sram_bank_conflict_free_matmul.cu -o sram_bank_conflict_free_matmul_benchmark

benchmark: naive_matmul_benchmark coalesced_matmul_benchmark sram_matmul_benchmark sram_bank_conflict_free_matmul_benchmark
	./naive_matmul_benchmark
	./coalesced_matmul_benchmark
	./sram_matmul_benchmark
	./sram_bank_conflict_free_matmul_benchmark

clean:
	rm -f naive_matmul_benchmark coalesced_matmul_benchmark sram_matmul_benchmark sram_bank_conflict_free_matmul_benchmark

.PHONY: all clean benchmark 