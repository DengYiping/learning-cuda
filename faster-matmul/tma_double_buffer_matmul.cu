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
    /* ---------------- Shared memory layout (double buffered + static mbarrier) -------------
       |  A_0 (dyn) |  A_1 (dyn) |  B_0 (dyn) |  B_1 (dyn) | + mbarrier[1] (static)
       -------------------------------------------------------------------------------------*/
    // Allocate shared memory for double-buffered tiles using dynamic shared memory
    alignas(128) extern __shared__ char smem_bytes[];

    // Statically allocate the mbarrier, ensuring 16-byte alignment.
    alignas(16) __shared__ uint64_t mbarrier_storage[1];
    // Get pointer to the mbarrier.
    uint64_t* mbar = mbarrier_storage;

    // Calculate the size needed for one stage's A and B tiles
    constexpr size_t smem_tile_A_bytes = BM * BK * sizeof(float);
    constexpr size_t smem_tile_B_bytes = BK * BN * sizeof(float);
    // Offset between the two buffers for A and B respectively (in bytes)
    constexpr size_t smem_offset_A = smem_tile_A_bytes;
    constexpr size_t smem_offset_B = smem_tile_B_bytes;
    // Total bytes for one stage (used for mbarrier arrive)
    constexpr size_t smem_one_stage_bytes = smem_tile_A_bytes + smem_tile_B_bytes;

    // Base pointers for the A and B double buffers in shared memory
    // Make non-const to allow swapping via XOR
    float* smem_base_A = reinterpret_cast<float*>(&smem_bytes[0]);
    // B buffers start after both A buffers
    float* smem_base_B = reinterpret_cast<float*>(&smem_bytes[2 * smem_tile_A_bytes]);

    // blockIdx.x is the block id in the N dimension, aka the column index of the block
    // blockIdx.y is the block id in the M dimension, aka the row index of the block

    // Each warp will calculate TM * TN elements
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Adjust C pointer for the current block's output
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    alignas(16) float thread_results[TM][TN] = {0.0f};
    // Double buffer registers for M and N tiles
    alignas(16) float reg_M[2][TM * 4];
    alignas(16) float reg_N[2][TN * 4];

    // ------------------------- PTX Asynchronous copy setup -------------------------
    auto block = cg::this_thread_block();
    const bool is_master_thread = (block.thread_rank() == 0);
    constexpr int THREADS_PER_BLOCK = BM * BN / (TM * TN);

    // Initialize the mbarrier from thread 0.
    if (is_master_thread) {
        ptx::mbarrier_init(mbar, THREADS_PER_BLOCK);
        // Fence to ensure initialization is visible to async copy units.
        ptx::fence_mbarrier_init_release_cluster();
    }
    // Sync all threads to ensure barrier is initialized before use.
    block.sync();

    // Phase variable for mbarrier wait cycles. Also used for read stage index.
    uint32_t phase = 0;
    uint32_t write_stage = 0;

    // Register stage indices for double buffering
    int reg_read_stage = 0;
    int reg_write_stage = 0;// will be toggled before first use

    // --- Prime the register buffer for dot_idx = 0 ---
    reg_write_stage ^= 1; // Now reg_write_stage = 1

    // Calculate total bytes per tile for A and B
    constexpr size_t total_bytes_per_stage = smem_one_stage_bytes; // Reuse calculation

    // Calculate base 2D offsets for this block
    const uint32_t base_offset_A_y = blockIdx.y * BM;
    const uint32_t base_offset_B_x = blockIdx.x * BN;

    // Offset for the next tile
    uint32_t next_offset_K = 0;

    // ---------------- Prime the pipeline : load the very first tile using TMA ----------------
    if (is_master_thread) {
        // Initiate A tile copy (TMA 2D) into the initial buffer (buffer 0)
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_base_A), // Use current base pointer
            (const uint64_t*)&tensor_map_A,
            next_offset_K, base_offset_A_y,
            mbar
        );
        // Initiate B tile copy (TMA 2D) into the initial buffer (buffer 0)
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_base_B), // Use current base pointer
            (const uint64_t*)&tensor_map_B,
            base_offset_B_x, next_offset_K,
            mbar
        );
        // Master thread arrives, indicating total expected bytes for this phase.
        ptx::mbarrier_arrive_expect_tx(mbar, total_bytes_per_stage);
    } else {
        // Other threads just arrive.
        ptx::mbarrier_arrive(mbar);
    }
    // Swap pointers to point to the other buffer (buffer 1) for the next copy
    smem_base_A = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(smem_base_A) ^ smem_offset_A);
    smem_base_B = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(smem_base_B) ^ smem_offset_B);

    // number of K-tiles we will iterate over
    const uint num_tiles = CEIL_DIV(K, BK);

    for (uint tile = 0; tile < num_tiles; ++tile) {
        next_offset_K += BK;
        // Wait for the copy initiated in the previous iteration (or priming phase) to complete.
        // This copy targeted the buffer *not* currently pointed to by smem_base_A/B.
        ptx::mbarrier_wait_parity(mbar, phase);

        // ---------------- Preload the next tile (if any) while computation continues ----------------
        if (tile + 1 < num_tiles) {
            if (is_master_thread) {
                 // Initiate A tile copy (TMA 2D) for the next tile into the buffer pointed to by current smem_base_A
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(smem_base_A), // Use current base pointer
                    (const uint64_t*)&tensor_map_A,
                    next_offset_K, base_offset_A_y,
                    mbar
                );
                // Initiate B tile copy (TMA 2D) for the next tile into the buffer pointed to by current smem_base_B
                 ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(smem_base_B), // Use current base pointer
                    (const uint64_t*)&tensor_map_B,
                    base_offset_B_x, next_offset_K,
                    mbar
                );
                // Master thread arrives, indicating total expected bytes for the next phase.
                ptx::mbarrier_arrive_expect_tx(mbar, total_bytes_per_stage);
                if (tile + 2 < num_tiles) {
                    ptx::prefetch_async_bulk_tensor_2d_global_l2(
                        (const uint64_t*)&tensor_map_A,
                        next_offset_K + BK,
                        base_offset_A_y
                    );
                    ptx::prefetch_async_bulk_tensor_2d_global_l2(
                        (const uint64_t*)&tensor_map_B,
                        base_offset_B_x,
                        next_offset_K + BK
                    );
                }
            } else {
                // Other threads just arrive for the next phase's copy.
                ptx::mbarrier_arrive(mbar);
            }
            // Note: Global A/B pointers are not advanced here as TensorMap handles addressing
        }

        // Swap pointers to point to the other buffer (buffer 1) for the next copy
        smem_base_A = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(smem_base_A) ^ smem_offset_A);
        smem_base_B = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(smem_base_B) ^ smem_offset_B);
        // ---------------- Matrix multiply on the current shared-memory tile ----------------

        #pragma unroll
        for (uint i = 0; i < TM; ++i) {
             reinterpret_cast<float4*>(&reg_M[reg_write_stage][i * 4])[0] = reinterpret_cast<const float4*>(&smem_base_A[(thread_row * TM + i) * BK])[0];
        }
        #pragma unroll
        for (uint j = 0; j < TN; j += 4) {
            reinterpret_cast<float4*>(&reg_N[reg_write_stage][j])[0] =
                 reinterpret_cast<const float4*>(&smem_base_B[0 * BN + (thread_col * TN + j)])[0];
            reinterpret_cast<float4*>(&reg_N[reg_write_stage][TN + j])[0] =
                 reinterpret_cast<const float4*>(&smem_base_B[1 * BN + (thread_col * TN + j)])[0];
            reinterpret_cast<float4*>(&reg_N[reg_write_stage][TN * 2 + j])[0] =
                 reinterpret_cast<const float4*>(&smem_base_B[2 * BN + (thread_col * TN + j)])[0];
            reinterpret_cast<float4*>(&reg_N[reg_write_stage][TN * 3 + j])[0] =
                 reinterpret_cast<const float4*>(&smem_base_B[3 * BN + (thread_col * TN + j)])[0];
        }

        #pragma unroll
        for (uint dot_idx = 0; dot_idx < BK / 4; ++dot_idx) {
            // Toggle stages for the next iteration's load / this iteration's compute
            reg_read_stage ^= 1;
            reg_write_stage ^= 1;

            const uint next_dot_idx = dot_idx + 1;
            if (next_dot_idx < BK / 4) {
                #pragma unroll
                for (uint i = 0; i < TM; ++i) {
                    reinterpret_cast<float4*>(&reg_M[reg_write_stage][i * 4])[0] = reinterpret_cast<const float4*>(&smem_base_A[(thread_row * TM + i) * BK + next_dot_idx * 4])[0];
                }
                #pragma unroll
                for (uint j = 0; j < TN; j += 4) {
                    reinterpret_cast<float4*>(&reg_N[reg_write_stage][j])[0] =
                        reinterpret_cast<const float4*>(&smem_base_B[(next_dot_idx * 4 + 0) * BN + (thread_col * TN + j)])[0];
                    reinterpret_cast<float4*>(&reg_N[reg_write_stage][TN + j])[0] =
                        reinterpret_cast<const float4*>(&smem_base_B[(next_dot_idx * 4 + 1) * BN + (thread_col * TN + j)])[0];
                    reinterpret_cast<float4*>(&reg_N[reg_write_stage][TN * 2 + j])[0] =
                        reinterpret_cast<const float4*>(&smem_base_B[(next_dot_idx * 4 + 2) * BN + (thread_col * TN + j)])[0];
                    reinterpret_cast<float4*>(&reg_N[reg_write_stage][TN * 3 + j])[0] =
                        reinterpret_cast<const float4*>(&smem_base_B[(next_dot_idx * 4 + 3) * BN + (thread_col * TN + j)])[0];
                }
            }

            #pragma unroll
            for (uint sub_dot_idx = 0; sub_dot_idx < 4; ++sub_dot_idx) {
                #pragma unroll
                for (uint i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (uint j = 0; j < TN; ++j) {
                        // Use the registers from the read stage for computation
                        thread_results[i][j] += reg_M[reg_read_stage][i * 4 + sub_dot_idx] * reg_N[reg_read_stage][j + sub_dot_idx * TN];
                    }
                }
            }
        }

        // Sync threads before next iteration to ensure computation is finished before
        // potentially overwriting the shared memory buffer in the next copy phase.
        // Also ensures all threads have updated phase/read_stage.
        block.sync();

        // Toggle read_stage to the buffer we just finished copying (and will read next)
        write_stage ^= 1;
        // Flip the phase for the next mbarrier wait.
        phase ^= 1;
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
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;

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

    // Shared memory: Calculate size needed for the double-buffered A/B tiles only.
    // Layout: A0 | A1 | B0 | B1
    size_t smem_bytes_dynamic = (2 * BM * BK + 2 * BN * BK) * sizeof(float);

    // Launch kernel with dynamic shared memory size for A/B buffers
    tma_double_buffered_matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim, smem_bytes_dynamic, stream>>>(
        tensor_map_A, tensor_map_B, d_C, m, n, k);
}

int main() {
    // Initialize CUDA Driver API
    CHECK_CUDA_DRIVER(cuInit(0));
    CUcontext ctx;
    CHECK_CUDA_DRIVER(cuCtxGetCurrent(&ctx)); // Ensure context exists

    // Use the best parameters found
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
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


    // Calculate required dynamic shared memory for A/B buffers
    size_t required_smem_dynamic = (2 * BM * BK + 2 * BN * BK) * sizeof(float); // Updated calculation
    std::cout << "Required dynamic shared memory per block: " << required_smem_dynamic << " bytes" << std::endl;
    // Calculate total static shared memory (mbarrier only)
    size_t required_smem_static = sizeof(uint64_t); // Size of the static mbarrier
    std::cout << "Required static shared memory per block: " << required_smem_static << " bytes" << std::endl;

    // Check if required dynamic shared memory exceeds limit
    if (required_smem_dynamic + required_smem_static > deviceProp.sharedMemPerMultiprocessor) {
         std::cerr << "Error: Required dynamic shared memory (" << required_smem_dynamic
                   << " bytes) plus static shared memory (" << required_smem_static
                   << " bytes) exceeds device limit per multiprocessor (" << deviceProp.sharedMemPerMultiprocessor
                   << " bytes)." << std::endl;
        return 1;
    }

    // Set shared memory carveout for this kernel - potentially higher if needed for TMA
    CHECK_CUDA(cudaFuncSetAttribute(
        tma_double_buffered_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100 // Max carveout, as TMA benefits from L1
    ));
    // Set dynamic shared memory size attribute for the kernel
    CHECK_CUDA(cudaFuncSetAttribute(
        (const void*)tma_double_buffered_matmul<BM, BN, BK, TM, TN>, // Need to cast kernel function pointer
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(required_smem_dynamic)
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