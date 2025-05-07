#include "faster_matmul.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_2d_block_tiling_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
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

    // -------------------------  Asynchronous copy helpers -------------------------
    constexpr auto scope = cuda::thread_scope_thread;
    auto thread = cooperative_groups::this_thread();
    auto block  = cooperative_groups::this_thread_block();
    cuda::pipeline<scope> pipe = cuda::make_pipeline();

    // Assume K is divisible by BK. Outer loop is over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        /* ------------------------------------------------------------------
           Load the next A/B tile from global memory to shared memory using
           cp.async (cuda::memcpy_async).  We keep exactly one copy of the
           tile in shared memory, therefore shared memory consumption is the
           same as the original kernel.
        ------------------------------------------------------------------ */

        pipe.producer_acquire();

        // load A_tile into shared_A
        for (int idx = threadIdx.x; idx < (BM * BK) / 4; idx += blockDim.x) {
            int row = idx / (BK / 4);       // which row within the tile
            int col4 = idx % (BK / 4);      // which float4 inside the row

            cuda::memcpy_async(
                thread,
                &reinterpret_cast<float4*>(shared_A)[idx],
                &reinterpret_cast<const float4*>(A)[row * (K / 4) + col4],
                cuda::aligned_size_t<16>(sizeof(float4)),
                pipe);
        }

        // load B_tile into shared_B
        for (int idx = threadIdx.x; idx < (BK * BN) / 4; idx += blockDim.x) {
            int row = idx / (BN / 4);
            int col4 = idx % (BN / 4);

            cuda::memcpy_async(
                thread,
                &reinterpret_cast<float4*>(shared_B)[idx],
                &reinterpret_cast<const float4*>(B)[row * (N / 4) + col4],
                cuda::aligned_size_t<16>(sizeof(float4)),
                pipe);
        }

        pipe.producer_commit();

        // Wait until the tile is fully available in shared memory
        pipe.consumer_wait();

        /* ------------------------------------------------------------------
           At this point the entire tile is in shared memory and can be
           consumed by all threads.  The copy from global memory was issued
           asynchronously;  with larger K-tiles the latency will largely be
           hidden behind the arithmetic below while we keep shared memory
           usage unchanged (no double buffering).
        ------------------------------------------------------------------ */

        // Advance global pointers so that they point at the *next* tile in K
        const float* next_A = A + BK;
        const float* next_B = B + BK * N;

        block.sync();
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

        block.sync();
        // We finished consuming this tile;  let the pipeline know.
        pipe.consumer_release();

        // Move on to the next tile in global memory
        A = next_A;
        B = next_B;
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
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int TM = 8;
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
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int TM = 8;
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

    std::cout << "Running async_global_to_shared_matmul benchmark:" << std::endl;

    // Run the benchmark with the naive matrix multiplication kernel
    float avg_time = run_benchmark<float>(
        launch_vectorized_2d_block_tiling_matmul, m, n, k
    );

    return 0;
}
