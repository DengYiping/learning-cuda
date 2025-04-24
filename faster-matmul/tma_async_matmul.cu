#include <cuda_runtime.h>
#include "faster_matmul.cuh"
#include <cooperative_groups.h>
#include "ptx.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
namespace cg = cooperative_groups;

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(BM * BN / (TM * TN)) void vectorized_2d_block_tiling_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    extern __shared__ float shared_A[];
    float* shared_B = shared_A + BM * BK;

    // blockIdx.x is the block id in the N dimension, aka the column index of the block
    // blockIdx.y is the block id in the M dimension, aka the row index of the block

    // Each warp will calculate 32 * TM * TN elements, with 32 being the columnar dim.
    // Num threads = BM * BN / (TM * TN), we will 2d tiling on the M, N dimension.
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Move blocktile to beginning of A's row and B's column
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float thread_results[TM][TN] = {0.0f};
    float reg_M[TM];
    float reg_N[TN];

    // ------------------------- PTX Asynchronous copy setup -------------------------
    auto block = cg::this_thread_block();
    const bool is_master_thread = (block.thread_rank() == 0);
    constexpr int THREADS_PER_BLOCK = BM * BN / (TM * TN);

    // Allocate shared memory for the mbarrier (only need 1 barrier).
    __shared__ uint64_t mbar[1];

    // Initialize the mbarrier from thread 0.
    if (is_master_thread) {
        ptx::mbarrier_init(&mbar[0], THREADS_PER_BLOCK);
        // Fence to ensure initialization is visible to async copy units.
        ptx::fence_mbarrier_init_release_cluster();
    }
    // Sync all threads to ensure barrier is initialized before use.
    block.sync();

    // Phase variable for mbarrier wait cycles.
    uint32_t phase = 0;

    // Assume K is divisible by BK. Outer loop is over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        /* ------------------------------------------------------------------
           Load the next A/B tile from global memory to shared memory using
           ptx::cp_async_bulk_tensor_1d_global_to_shared.
        ------------------------------------------------------------------ */

        // Calculate total bytes to be copied in this iteration.
        constexpr size_t bytes_A_tile = BM * BK * sizeof(float);
        constexpr size_t bytes_B_tile = BK * BN * sizeof(float);
        constexpr size_t total_bytes = bytes_A_tile + bytes_B_tile;

        // Master thread initiates all copies and arrives once with expect_tx.
        if (is_master_thread) {
            // Initiate A tile copy (row by row)
            #pragma unroll
            for (uint i = 0; i < BM; i++) {
                constexpr size_t bytes_to_copy = BK * sizeof(float);
                 ptx::cp_async_bulk_tensor_1d_global_to_shared(
                    reinterpret_cast<uint64_t*>(shared_A + i * BK),
                    reinterpret_cast<const uint64_t*>(A + i * K),
                    bytes_to_copy,
                    &mbar[0]
                );
            }
            // Initiate B tile copy (row by row)
            #pragma unroll
             for (uint i = 0; i < BK; i++) {
                 constexpr size_t bytes_to_copy = BN * sizeof(float);
                 ptx::cp_async_bulk_tensor_1d_global_to_shared(
                    reinterpret_cast<uint64_t*>(shared_B + i * BN),
                    reinterpret_cast<const uint64_t*>(B + i * N),
                    bytes_to_copy,
                    &mbar[0]
                );
            }
            // Master thread arrives, indicating total expected bytes for this phase.
            ptx::mbarrier_arrive_expect_tx(&mbar[0], total_bytes);
        } else {
            // Other threads just arrive.
            ptx::mbarrier_arrive(&mbar[0]);
        }

        // Wait for all threads to arrive and for the async copies (total_bytes) to complete.
        ptx::mbarrier_wait_parity(&mbar[0], phase);

        /* ------------------------------------------------------------------
           At this point the entire tile is in shared memory and can be
           consumed by all threads.
        ------------------------------------------------------------------ */

        // Perform matrix multiplication for this tile
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load one column of A and one row of B from shared memory into
            // registers and compute a TM x TN outer product.

            for (uint i = 0; i < TM; ++i) {
                reg_M[i] = shared_A[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint j = 0; j < TN; j += 4) {
                reinterpret_cast<float4*>(&reg_N[j])[0] =
                    reinterpret_cast<float4*>(&shared_B[dot_idx * BN + (thread_col * TN + j)])[0];
            }

            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_M[i] * reg_N[j];
                }
            }
        }
        // Sync threads before starting next K-tile iteration (ensures shared memory is ready for next load).
        block.sync();

        // Move on to the next tile in global memory
        A += BK;
        B += BK * N;

        // Flip the phase for the next mbarrier wait.
        phase ^= 1;
    }

    // Store the results
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j+= 4) {
            reinterpret_cast<float4*>(&C[(thread_row * TM + i) * N + (thread_col * TN + j)])[0] = reinterpret_cast<float4*>(&thread_results[i][j])[0];
        }
    }
}

// Kernel launcher function
void launch_vectorized_2d_block_tiling_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int m, int n, int k, cudaStream_t stream) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int TM = 4;
    constexpr int TN = 4;

    // Each thread will calculate TM * TN elements
    dim3 blockDim(BM * BN / (TM * TN)); 
    // Reversing order to optimize L2 cache access. Grid will move on the N dimension fast and M dimension slow.
    // With row-major layout, this is more cache-friendly.
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    
    vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim, BM * BK * sizeof(float) + BN * BK * sizeof(float), stream>>>(d_A, d_B, d_C, m, n, k);
}




int main() {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int TM = 4;
    constexpr int TN = 4;

    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Shared memory size: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;


    // Set shared memory carveout for this kernel
    CHECK_CUDA(cudaFuncSetAttribute(
        vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        75
    ));
    // Set shared memory size for this kernel
    CHECK_CUDA(cudaFuncSetAttribute(
        vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        128 * 1024
    ));

    // Default matrix dimensions
    int m = 4096; // Matrix A: m x k
    int n = 2048; // Matrix B: k x n, Matrix C: m x n
    int k = 512;

    std::cout << "Running Vectorized 2D block tiling matrix multiplication benchmark:" << std::endl;

    // Run the benchmark with the naive matrix multiplication kernel
    float avg_time = run_benchmark<float>(
        launch_vectorized_2d_block_tiling_matmul, m, n, k
    );

    return 0;
}