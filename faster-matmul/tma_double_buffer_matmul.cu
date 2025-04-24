#include <cuda_runtime.h>
#include "faster_matmul.cuh"
#include <cooperative_groups.h>
#include "ptx.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
namespace cg = cooperative_groups;

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(BM * BN / (TM * TN)) void tma_double_buffered_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    /* ---------------- Shared memory layout (double buffered) -------------
       |  A_0  |  B_0  |  A_1  |  B_1  | + mbarrier[1]
       --------------------------------------------------------------------*/
    // Allocate shared memory with space for the mbarrier at the end.
    extern __shared__ uint8_t smem_bytes[];
    // Ensure 16-byte alignment for mbarrier.
    constexpr size_t smem_tile_bytes = (BM * BK + BN * BK) * sizeof(float);
    constexpr size_t mbarrier_offset = 2 * smem_tile_bytes;
    constexpr size_t mbarrier_aligned_offset = (mbarrier_offset + 15) & ~15; // Align to 16 bytes
    uint64_t* mbar = reinterpret_cast<uint64_t*>(&smem_bytes[mbarrier_aligned_offset]);

    float* smem_A[2] = { reinterpret_cast<float*>(&smem_bytes[0]),
                         reinterpret_cast<float*>(&smem_bytes[smem_tile_bytes]) };
    float* smem_B[2] = { reinterpret_cast<float*>(&smem_bytes[BM * BK * sizeof(float)]),
                         reinterpret_cast<float*>(&smem_bytes[smem_tile_bytes + BM * BK * sizeof(float)]) };

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

    // Stage indices (ping-pong)
    int write_stage = 0;
    int read_stage = 0;

    // Calculate total bytes per tile for A and B
    constexpr size_t bytes_A_tile = BM * BK * sizeof(float);
    constexpr size_t bytes_B_tile = BK * BN * sizeof(float);
    constexpr size_t total_bytes_per_stage = bytes_A_tile + bytes_B_tile;

    // ---------------- Prime the pipeline : load the very first tile using TMA ----------------
    if (is_master_thread) {
        // Initiate A tile copy (row by row)
        #pragma unroll
        for (uint i = 0; i < BM; i++) {
            constexpr size_t bytes_to_copy = BK * sizeof(float);
             ptx::cp_async_bulk_tensor_1d_global_to_shared(
                reinterpret_cast<uint64_t*>(smem_A[write_stage] + i * BK),
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
                reinterpret_cast<uint64_t*>(smem_B[write_stage] + i * BN),
                reinterpret_cast<const uint64_t*>(B + i * N),
                bytes_to_copy,
                &mbar[0]
            );
        }
        // Master thread arrives, indicating total expected bytes for this phase.
        ptx::mbarrier_arrive_expect_tx(&mbar[0], total_bytes_per_stage);
    } else {
        // Other threads just arrive.
        ptx::mbarrier_arrive(&mbar[0]);
    }

    // Advance global pointers to the next tile along K
    A += BK;            // next A tile starts BK columns to the right
    B += BK * N;        // next B tile is BK rows down

    // number of K-tiles we will iterate over
    const uint num_tiles = CEIL_DIV(K, BK);

    for (uint tile = 0; tile < num_tiles; ++tile) {
        // Wait for the copy initiated in the previous iteration (or priming phase) to complete.
        ptx::mbarrier_wait_parity(&mbar[0], phase);

        // ---------------- Preload the next tile (if any) while computation continues ----------------
        if (tile + 1 < num_tiles) {
            write_stage ^= 1; // toggle between 0 and 1
            if (is_master_thread) {
                 // Initiate A tile copy (row by row)
                #pragma unroll
                for (uint i = 0; i < BM; i++) {
                    constexpr size_t bytes_to_copy = BK * sizeof(float);
                     ptx::cp_async_bulk_tensor_1d_global_to_shared(
                        reinterpret_cast<uint64_t*>(smem_A[write_stage] + i * BK),
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
                        reinterpret_cast<uint64_t*>(smem_B[write_stage] + i * BN),
                        reinterpret_cast<const uint64_t*>(B + i * N),
                        bytes_to_copy,
                        &mbar[0]
                    );
                }
                // Master thread arrives, indicating total expected bytes for the next phase.
                ptx::mbarrier_arrive_expect_tx(&mbar[0], total_bytes_per_stage);
            } else {
                // Other threads just arrive for the next phase's copy.
                ptx::mbarrier_arrive(&mbar[0]);
            }

            // Advance global pointers for A and B to point at the subsequent tile
            A += BK;
            B += BK * N;
        }

        // Set up shared pointers for the tile we are about to compute on
        const float* shared_A = smem_A[read_stage];
        const float* shared_B = smem_B[read_stage];

        // ---------------- Matrix multiply on the current shared-memory tile ----------------
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load one element from A tile into registers per thread
            for (uint i = 0; i < TM; ++i) {
                reg_M[i] = shared_A[(thread_row * TM + i) * BK + dot_idx];
            }
            // Load TN elements from B tile into registers per thread (vectorised in chunks of 4)
            for (uint j = 0; j < TN; j += 4) {
                reinterpret_cast<float4*>(&reg_N[j])[0] =
                    reinterpret_cast<const float4*>(&shared_B[dot_idx * BN + (thread_col * TN + j)])[0];
            }
            // FMA accumulation into thread-private result tile
            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_M[i] * reg_N[j];
                }
            }
        }

        // Toggle read_stage to the buffer we just finished copying (and will read next)
        read_stage ^= 1;
        // Flip the phase for the next mbarrier wait.
        phase ^= 1;
        // Sync threads before next iteration to ensure computation is finished before
        // potentially overwriting the shared memory buffer in the next copy phase.
        // Also ensures all threads have updated phase/read_stage.
        block.sync();
    }

    // At this point, all tiles have been processed and results are in thread_results.
    // ---------------- Store the results to global memory ----------------
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; j += 4) {
            reinterpret_cast<float4*>(&C[(thread_row * TM + i) * N + (thread_col * TN + j)])[0] =
                reinterpret_cast<float4*>(&thread_results[i][j])[0];
        }
    }
}

// Kernel launcher function
void launch_tma_double_buffered_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int m, int n, int k, cudaStream_t stream) {
    // Use the same parameters as double_buffering_matmul for comparison
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Each thread will calculate TM * TN elements
    dim3 blockDim(BM * BN / (TM * TN));
    // Reversing order to optimize L2 cache access. Grid will move on the N dimension fast and M dimension slow.
    // With row-major layout, this is more cache-friendly.
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    // Shared memory: 2 stages for A and B tiles + space for mbarrier (aligned)
    size_t smem_tile_bytes = (BM * BK + BN * BK) * sizeof(float);
    size_t mbarrier_offset = 2 * smem_tile_bytes;
    size_t mbarrier_aligned_offset = (mbarrier_offset + 15) & ~15; // Align to 16 bytes
    size_t smem_bytes = mbarrier_aligned_offset + sizeof(uint64_t); // Total size needed

    tma_double_buffered_matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim, smem_bytes, stream>>>(d_A, d_B, d_C, m, n, k);
}

int main() {
    // Use the same parameters as double_buffering_matmul for comparison
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int TM = 8;
    constexpr int TN = 8;

    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Shared memory size: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;

    // Check if device supports cooperative launch
    int cooperativeLaunch;
    CHECK_CUDA(cudaDeviceGetAttribute(&cooperativeLaunch, cudaDevAttrCooperativeLaunch, 0));
    if (!cooperativeLaunch) {
        std::cerr << "Device does not support cooperative launch, which is required for TMA mbarrier." << std::endl;
        //return 1; // Or handle gracefully
    }
     // Check if device supports cluster launch needed for mbarrier
    int clusterLaunch;
    CHECK_CUDA(cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, 0));
    if (!clusterLaunch) {
        std::cerr << "Device does not support cluster launch, which is required for TMA mbarrier." << std::endl;
        // return 1; // Or handle gracefully
    }
     // Check if device supports asynchronous copy specifically
    int asyncEngineCount;
    CHECK_CUDA(cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, 0));
    if (asyncEngineCount == 0) {
        std::cerr << "Device does not have async copy engines required for TMA." << std::endl;
       // return 1; // Or handle gracefully
    }

     std::cout << "Device supports cooperative launch: " << (cooperativeLaunch ? "Yes" : "No") << std::endl;
     std::cout << "Device supports cluster launch: " << (clusterLaunch ? "Yes" : "No") << std::endl;
     std::cout << "Device async engine count: " << asyncEngineCount << std::endl;


    // Calculate required shared memory
    size_t smem_tile_bytes = (BM * BK + BN * BK) * sizeof(float);
    size_t mbarrier_offset = 2 * smem_tile_bytes;
    size_t mbarrier_aligned_offset = (mbarrier_offset + 15) & ~15;
    size_t required_smem = mbarrier_aligned_offset + sizeof(uint64_t);
    std::cout << "Required shared memory per block: " << required_smem << " bytes" << std::endl;
    if (required_smem > deviceProp.sharedMemPerMultiprocessor) {
         std::cerr << "Error: Required shared memory (" << required_smem
                   << " bytes) exceeds device limit (" << deviceProp.sharedMemPerMultiprocessor
                   << " bytes)." << std::endl;
        return 1;
    }

    // Set shared memory carveout for this kernel - potentially higher if needed for TMA
    CHECK_CUDA(cudaFuncSetAttribute(
        tma_double_buffered_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100 // Max carveout, as TMA benefits from L1
    ));
    // Set dynamic shared memory size for this kernel
    CHECK_CUDA(cudaFuncSetAttribute(
        tma_double_buffered_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(required_smem)
    ));

    // Default matrix dimensions
    int m = 4096; // Matrix A: m x k
    int n = 2048; // Matrix B: k x n, Matrix C: m x n
    int k = 512;

    std::cout << "Running TMA Double Buffered matrix multiplication benchmark:" << std::endl;

    // Run the benchmark with the new kernel
    float avg_time = run_benchmark<float>(
        launch_tma_double_buffered_matmul, m, n, k
    );

    return 0;
} 