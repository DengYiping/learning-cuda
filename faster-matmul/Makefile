NVCC = nvcc
NVCCFLAGS = -O3 -std=c++20 -arch=sm_90 -Xcompiler -Wall,-Wextra --generate-line-info

all: naive_matmul coalesced_matmul sram_matmul 1d_block_tiling_matmul 2d_block_tiling_matmul vectorized_2d_block_tiling_matmul double_buffering_matmul

naive_matmul: naive_matmul.cu
	$(NVCC) $(NVCCFLAGS) naive_matmul.cu -o naive_matmul

naive_matmul_profiling: naive_matmul
	ncu -f -o naive_matmul_report --set full ./naive_matmul

coalesced_matmul: coalesced_matmul.cu
	$(NVCC) $(NVCCFLAGS) coalesced_matmul.cu -o coalesced_matmul

coalesced_matmul_profiling: coalesced_matmul
	ncu -f -o coalesced_matmul_report --set full ./coalesced_matmul

sram_matmul: sram_matmul.cu
	$(NVCC) $(NVCCFLAGS) sram_matmul.cu -o sram_matmul

sram_matmul_profiling: sram_matmul
	ncu -f -o sram_matmul_report --set full ./sram_matmul

1d_block_tiling_matmul: 1d_block_tiling_matmul.cu
	$(NVCC) $(NVCCFLAGS) 1d_block_tiling_matmul.cu -o 1d_block_tiling_matmul

1d_block_tiling_matmul_profiling: 1d_block_tiling_matmul
	ncu -f -o 1d_block_tiling_matmul_report --set full ./1d_block_tiling_matmul

2d_block_tiling_matmul: 2d_block_tiling_matmul.cu
	$(NVCC) $(NVCCFLAGS) 2d_block_tiling_matmul.cu -o 2d_block_tiling_matmul

2d_block_tiling_matmul_profiling: 2d_block_tiling_matmul
	ncu -f -o 2d_block_tiling_matmul_report --set full ./2d_block_tiling_matmul

vectorized_2d_block_tiling_matmul: vectorized_2d_block_tiling_matmul.cu
	$(NVCC) $(NVCCFLAGS) vectorized_2d_block_tiling_matmul.cu -o vectorized_2d_block_tiling_matmul

vectorized_2d_block_tiling_matmul_profiling: vectorized_2d_block_tiling_matmul
	ncu -f -o vectorized_2d_block_tiling_matmul_report --set full ./vectorized_2d_block_tiling_matmul

double_buffering_matmul: double_buffering_matmul.cu
	$(NVCC) $(NVCCFLAGS) double_buffering_matmul.cu -o double_buffering_matmul

double_buffering_matmul_profiling: double_buffering_matmul
	ncu -f -o double_buffering_matmul_report --set full ./double_buffering_matmul

clean:
	rm -f naive_matmul coalesced_matmul sram_matmul 1d_block_tiling_matmul 2d_block_tiling_matmul vectorized_2d_block_tiling_matmul double_buffering_matmul
	rm -f naive_matmul_report.ncu-rep coalesced_matmul_report.ncu-rep sram_matmul_report.ncu-rep 1d_block_tiling_matmul_report.ncu-rep 2d_block_tiling_matmul_report.ncu-rep vectorized_2d_block_tiling_matmul_report.ncu-rep double_buffering_matmul_report.ncu-rep

.PHONY: all clean benchmark