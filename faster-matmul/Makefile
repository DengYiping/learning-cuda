NVCC = nvcc
NVCCFLAGS = -O3 -std=c++20 -gencode=arch=compute_90,code=sm_90 -Xcompiler -Wall,-Wextra --generate-line-info -lcuda --expt-relaxed-constexpr

all: naive_matmul coalesced_matmul sram_matmul 1d_block_tiling_matmul 2d_block_tiling_matmul vectorized_2d_block_tiling_matmul double_buffering_matmul async_global_to_shared_matmul tma_double_buffer_matmul sync_double_buffer_matmul sync_db_warp_matmul tma_async_matmul_wt tma_double_buffer_matmul_wt

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

async_global_to_shared_matmul: async_global_to_shared_matmul.cu
	$(NVCC) $(NVCCFLAGS) async_global_to_shared_matmul.cu -o async_global_to_shared_matmul

async_global_to_shared_matmul_profiling: async_global_to_shared_matmul
	ncu -f -o async_global_to_shared_matmul_report --set full ./async_global_to_shared_matmul

tma_async_matmul_wt: tma_async_matmul_wt.cu
	$(NVCC) $(NVCCFLAGS) tma_async_matmul_wt.cu -o tma_async_matmul_wt

tma_async_matmul_wt_profiling: tma_async_matmul_wt
	ncu -f -o tma_async_matmul_wt_report --set full ./tma_async_matmul_wt

tma_double_buffer_matmul: tma_double_buffer_matmul.cu
	$(NVCC) $(NVCCFLAGS) tma_double_buffer_matmul.cu -o tma_double_buffer_matmul

tma_double_buffer_matmul_profiling: tma_double_buffer_matmul
	ncu -f -o tma_double_buffer_matmul_report --set full ./tma_double_buffer_matmul

tma_double_buffer_matmul_wt: tma_double_buffer_matmul_wt.cu
	$(NVCC) $(NVCCFLAGS) tma_double_buffer_matmul_wt.cu -o tma_double_buffer_matmul_wt

tma_double_buffer_matmul_wt_profiling: tma_double_buffer_matmul_wt
	ncu -f -o tma_double_buffer_matmul_wt_report --set full ./tma_double_buffer_matmul_wt

sync_double_buffer_matmul: sync_double_buffer_matmul.cu
	$(NVCC) $(NVCCFLAGS) sync_double_buffer_matmul.cu -o sync_double_buffer_matmul

sync_double_buffer_matmul_profiling: sync_double_buffer_matmul
	ncu -f -o sync_double_buffer_matmul_report --set full ./sync_double_buffer_matmul

sync_db_warp_matmul: sync_db_warp_matmul.cu
	$(NVCC) $(NVCCFLAGS) sync_db_warp_matmul.cu -o sync_db_warp_matmul

sync_db_warp_matmul_profiling: sync_db_warp_matmul
	ncu -f -o sync_db_warp_matmul_report --set full ./sync_db_warp_matmul

clean:
	rm -f naive_matmul coalesced_matmul sram_matmul 1d_block_tiling_matmul 2d_block_tiling_matmul vectorized_2d_block_tiling_matmul double_buffering_matmul async_global_to_shared_matmul tma_double_buffer_matmul sync_double_buffer_matmul sync_db_warp_matmul tma_async_matmul_wt tma_double_buffer_matmul_wt

	rm -f naive_matmul_report.ncu-rep coalesced_matmul_report.ncu-rep sram_matmul_report.ncu-rep 1d_block_tiling_matmul_report.ncu-rep 2d_block_tiling_matmul_report.ncu-rep vectorized_2d_block_tiling_matmul_report.ncu-rep double_buffering_matmul_report.ncu-rep async_global_to_shared_matmul_report.ncu-rep tma_double_buffer_matmul_report.ncu-rep sync_double_buffer_matmul_report.ncu-rep sync_db_warp_matmul_report.ncu-rep tma_async_matmul_wt_report.ncu-rep tma_double_buffer_matmul_wt_report.ncu-rep

run: all
	echo "Running all kernels..."
	./naive_matmul
	./coalesced_matmul
	./sram_matmul
	./1d_block_tiling_matmul
	./2d_block_tiling_matmul
	./vectorized_2d_block_tiling_matmul
	./double_buffering_matmul
	./async_global_to_shared_matmul
	./tma_async_matmul_wt
	./tma_double_buffer_matmul
	./tma_double_buffer_matmul_wt
	./sync_double_buffer_matmul
	./sync_db_warp_matmul

.PHONY: all clean benchmark run

