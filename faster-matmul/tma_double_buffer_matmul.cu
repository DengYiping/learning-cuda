#include <cuda_runtime.h>
#include "faster_matmul.cuh"
#include <cooperative_groups.h>
#include "ptx.cuh"
#include <cuda.h> // Include CUDA Driver API header
#include <stdio.h> // For printf debugging
#include <cstdlib> // For exit()

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
namespace cg = cooperative_groups;

// Helper macro for CUDA Driver API error checking
#define CHECK_CUDA_DRIVER(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *err_str; \
        cuGetErrorString(err, &err_str); \
        fprintf(stderr, "CUDA Driver error in %s at line %d: %s\n", __FILE__, __LINE__, err_str); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(BM * BN / (TM * TN)) void tma_double_buffered_matmul(
    const __grid_constant__ CUtensorMap tensor_map_A, // Pass by value with __grid_constant__
    const __grid_constant__ CUtensorMap tensor_map_B, // Pass by value with __grid_constant__
    float* __restrict__ C,
    const int M, const int N, const int K)
{
    /* ---------------- Shared memory layout (double buffered) -------------
       |  A_0  |  B_0  |  A_1  |  B_1  | + mbarrier[1]
       --------------------------------------------------------------------*/
    // Allocate shared memory with space for the mbarrier at the end.
    // Use char for byte-level access and ensure alignment
    alignas(128) extern __shared__ char smem_bytes[];
    // Ensure 16-byte alignment for mbarrier relative to the start of smem_bytes.
    constexpr size_t smem_tile_bytes = (BM * BK + BN * BK) * sizeof(float);
    constexpr size_t mbarrier_offset = 2 * smem_tile_bytes;
    constexpr size_t mbarrier_aligned_offset = (mbarrier_offset + 15) & ~15; // Align to 16 bytes
    uint64_t* mbar = reinterpret_cast<uint64_t*>(&smem_bytes[mbarrier_aligned_offset]);

    // Derive float pointers from the aligned char buffer
    float* smem_A[2] = { reinterpret_cast<float*>(&smem_bytes[0]),
                         reinterpret_cast<float*>(&smem_bytes[smem_tile_bytes]) };
    float* smem_B[2] = { reinterpret_cast<float*>(&smem_bytes[BM * BK * sizeof(float)]),
                         reinterpret_cast<float*>(&smem_bytes[smem_tile_bytes + BM * BK * sizeof(float)]) };

    // blockIdx.x is the block id in the N dimension, aka the column index of the block
    // blockIdx.y is the block id in the M dimension, aka the row index of the block

    // Each warp will calculate TM * TN elements
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Adjust C pointer for the current block's output
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    alignas(16) float thread_results[TM][TN] = {0.0f};
    alignas(16) float reg_M[TM];
    alignas(16) float reg_N[TN];

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

    // Calculate base 2D offsets for this block
    const uint32_t base_offset_A_y = blockIdx.y * BM;
    const uint32_t base_offset_B_x = blockIdx.x * BN;

    // ---------------- Prime the pipeline : load the very first tile using TMA ----------------
    if (is_master_thread) {
        const uint32_t current_offset_A_x = 0; // First K-tile
        const uint32_t current_offset_B_y = 0; // First K-tile

        // Initiate A tile copy (TMA 2D)
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_A[write_stage]),
            (const uint64_t*)&tensor_map_A,
            current_offset_A_x, base_offset_A_y,
            &mbar[0]
        );
        // Initiate B tile copy (TMA 2D)
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_B[write_stage]),
            (const uint64_t*)&tensor_map_B,
            base_offset_B_x, current_offset_B_y,
            &mbar[0]
        );
        // Master thread arrives, indicating total expected bytes for this phase.
        ptx::mbarrier_arrive_expect_tx(&mbar[0], total_bytes_per_stage);
    } else {
        // Other threads just arrive.
        ptx::mbarrier_arrive(&mbar[0]);
    }

    // number of K-tiles we will iterate over
    const uint num_tiles = CEIL_DIV(K, BK);

    for (uint tile = 0; tile < num_tiles; ++tile) {
        // Wait for the copy initiated in the previous iteration (or priming phase) to complete.
        ptx::mbarrier_wait_parity(&mbar[0], phase);

        // ---------------- Preload the next tile (if any) while computation continues ----------------
        if (tile + 1 < num_tiles) {
            write_stage ^= 1; // toggle between 0 and 1
            if (is_master_thread) {
                // Calculate offsets for the *next* tile
                const uint32_t next_offset_A_x = (tile + 1) * BK;
                const uint32_t next_offset_B_y = (tile + 1) * BK;

                 // Initiate A tile copy (TMA 2D) for the next tile
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(smem_A[write_stage]),
                    (const uint64_t*)&tensor_map_A,
                    next_offset_A_x, base_offset_A_y,
                    &mbar[0]
                );
                // Initiate B tile copy (TMA 2D) for the next tile
                 ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(smem_B[write_stage]),
                    (const uint64_t*)&tensor_map_B,
                    base_offset_B_x, next_offset_B_y,
                    &mbar[0]
                );
                // Master thread arrives, indicating total expected bytes for the next phase.
                ptx::mbarrier_arrive_expect_tx(&mbar[0], total_bytes_per_stage);
            } else {
                // Other threads just arrive for the next phase's copy.
                ptx::mbarrier_arrive(&mbar[0]);
            }
            // Note: Global A/B pointers are not advanced here as TensorMap handles addressing
        }

        // Set up shared pointers for the tile we are about to compute on
        const float* shared_A = smem_A[read_stage];
        const float* shared_B = smem_B[read_stage];

        // ---------------- Matrix multiply on the current shared-memory tile ----------------
        #pragma unroll
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load one element from A tile into registers per thread
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                reg_M[i] = shared_A[(thread_row * TM + i) * BK + dot_idx];
            }
            // Load TN elements from B tile into registers per thread (vectorised in chunks of 4)
            #pragma unroll
            for (uint j = 0; j < TN; j += 4) {
                reinterpret_cast<float4*>(&reg_N[j])[0] =
                    reinterpret_cast<const float4*>(&shared_B[dot_idx * BN + (thread_col * TN + j)])[0];
            }
            // FMA accumulation into thread-private result tile
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                #pragma unroll
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
    #pragma unroll
    for (uint i = 0; i < TM; ++i) {
        #pragma unroll
        for (uint j = 0; j < TN; j += 4) {
            uint write_idx_base = (thread_row * TM + i) * N + (thread_col * TN + j);
            reinterpret_cast<float4*>(&C[write_idx_base])[0] =
                reinterpret_cast<float4*>(&thread_results[i][j])[0];
        }
    }
}

// Kernel launcher function
void launch_tma_double_buffered_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int m, int n, int k, cudaStream_t stream) {
    // Use the best parameters found
    constexpr int BM = 64;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 4;

    // Create Tensor Maps
    CUtensorMap tensor_map_A;
    CUtensorMap tensor_map_B;

    const cuuint32_t elementStrides[] = {1, 1}; // Contiguous access

    // Tensor A (M x K) -> {inner (K), outer (M)}
    const uint64_t globalDimA[] = {(uint64_t)k, (uint64_t)m};
    // Global stride for A: stride between rows = K * sizeof(float)
    const uint64_t globalStrideA[] = {sizeof(float), (uint64_t)k * sizeof(float)};
    const cuuint32_t boxDimA[] = {BK, BM}; // {inner dim size, outer dim size}

    CHECK_CUDA_DRIVER(cuTensorMapEncodeTiled(
        &tensor_map_A,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                          // rank
        (void*)d_A,                 // globalAddress
        globalDimA,                 // globalDim
        globalStrideA + 1,          // globalStride (expects outer stride only)
        boxDimA,                    // boxDim
        elementStrides,             // elementStride
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    // Tensor B (K x N) -> {inner (N), outer (K)}
    const uint64_t globalDimB[] = {(uint64_t)n, (uint64_t)k};
    // Global stride for B: stride between rows = N * sizeof(float)
    const uint64_t globalStrideB[] = {sizeof(float), (uint64_t)n * sizeof(float)};
    const cuuint32_t boxDimB[] = {BN, BK}; // {inner dim size, outer dim size}

     CHECK_CUDA_DRIVER(cuTensorMapEncodeTiled(
        &tensor_map_B,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                          // rank
        (void*)d_B,                 // globalAddress
        globalDimB,                 // globalDim
        globalStrideB + 1,          // globalStride (expects outer stride only)
        boxDimB,                    // boxDim
        elementStrides,             // elementStride
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));


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

    tma_double_buffered_matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim, smem_bytes, stream>>>(
        tensor_map_A, tensor_map_B, d_C, m, n, k);
}

int main() {
    // Initialize CUDA Driver API
    CHECK_CUDA_DRIVER(cuInit(0));
    CUcontext ctx;
    CHECK_CUDA_DRIVER(cuCtxGetCurrent(&ctx)); // Ensure context exists

    // Use the best parameters found
    constexpr int BM = 64;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 4;

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
        (const void*)tma_double_buffered_matmul<BM, BN, BK, TM, TN>, // Need to cast kernel function pointer
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