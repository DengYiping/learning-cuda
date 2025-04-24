#include <cuda_runtime.h>
#include "faster_matmul.cuh"
#include <cooperative_groups.h>
#include "ptx.cuh"
#include <cuda.h> // Include CUDA Driver API header
#include <stdio.h> // For printf debugging

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
__global__ __launch_bounds__(BM * BN / (TM * TN)) void vectorized_2d_block_tiling_matmul(
    const __grid_constant__ CUtensorMap tensor_map_A, // Pass by value with __grid_constant__
    const __grid_constant__ CUtensorMap tensor_map_B, // Pass by value with __grid_constant__
    float* __restrict__ C,
    const int M, const int N, const int K)
{
    // Define aligned shared memory
    alignas(128) extern __shared__ char smem_buffer[]; // Use char for byte-level layout, align to 16 bytes

    // Derive float pointers from the aligned buffer
    float* shared_A = reinterpret_cast<float*>(smem_buffer);
    float* shared_B = shared_A + BM * BK;

    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Adjust C pointer for the current block
    float* C_block_start = C + blockIdx.y * BM * N + blockIdx.x * BN;

    alignas(16) float thread_results[TM][TN] = {0.0f};
    alignas(16) float reg_M[TM];
    alignas(16) float reg_N[TN];

    // PTX Asynchronous copy setup
    auto block = cg::this_thread_block();
    const bool is_master_thread = (block.thread_rank() == 0);
    constexpr int THREADS_PER_BLOCK = BM * BN / (TM * TN);

    alignas(8) __shared__ uint64_t mbar[1];

    if (is_master_thread) {
        ptx::mbarrier_init(&mbar[0], THREADS_PER_BLOCK);
        ptx::fence_mbarrier_init_release_cluster(); // Ensure init is visible to async units
    }
    block.sync(); // Ensure barrier is initialized before use

    uint32_t phase = 0; // Phase for mbarrier wait cycles

    // Outer loop over block tiles in K dimension
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load the next A/B tile from global memory to shared memory using TMA
        constexpr size_t bytes_A_tile = BM * BK * sizeof(float);
        constexpr size_t bytes_B_tile = BK * BN * sizeof(float);
        constexpr size_t total_bytes = bytes_A_tile + bytes_B_tile;

        // Calculate 2D offsets for TMA
        const uint32_t offset_A_x = bkIdx;
        const uint32_t offset_A_y = blockIdx.y * BM;
        const uint32_t offset_B_x = blockIdx.x * BN;
        const uint32_t offset_B_y = bkIdx;

        if (is_master_thread) {
            // Initiate A tile copy (TMA)
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(shared_A),
                (const uint64_t*)&tensor_map_A,
                offset_A_x, offset_A_y,
                &mbar[0]
            );
            // Initiate B tile copy (TMA)
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(shared_B),
                (const uint64_t*)&tensor_map_B,
                offset_B_x, offset_B_y,
                &mbar[0]
            );
            ptx::mbarrier_arrive_expect_tx(&mbar[0], total_bytes);
        } else {
            ptx::mbarrier_arrive(&mbar[0]);
        }

        // Wait for TMA copies to complete
        ptx::mbarrier_wait_parity(&mbar[0], phase);

        // Perform matrix multiplication for this tile
        #pragma unroll
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load column of A and row of B into registers
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                reg_M[i] = shared_A[(thread_row * TM + i) * BK + dot_idx];
            }
            #pragma unroll
            for (uint j = 0; j < TN; j += 4) {
                reinterpret_cast<float4*>(&reg_N[j])[0] =
                    reinterpret_cast<const float4*>(&shared_B[dot_idx * BN + (thread_col * TN + j)])[0];
            }

            // Compute outer product and accumulate
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                #pragma unroll
                for (uint j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_M[i] * reg_N[j];
                }
            }
        }

        block.sync(); // Ensure shared mem reads finish before next K-tile TMA overwrite

        phase ^= 1; // Flip phase for next mbarrier wait
    }

    // Store the results
    #pragma unroll
    for (uint i = 0; i < TM; i++) {
        #pragma unroll
        for (uint j = 0; j < TN; j+= 4) {
             uint write_idx_base = (thread_row * TM + i) * N + (thread_col * TN + j);
             reinterpret_cast<float4*>(&C_block_start[write_idx_base])[0] = reinterpret_cast<float4*>(&thread_results[i][j])[0];
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

    // Create Tensor Maps
    CUtensorMap tensor_map_A;
    CUtensorMap tensor_map_B;

    const cuuint32_t elementStrides[] = {1, 1}; // Contiguous access

    // Tensor A (M x K) -> {inner (K), outer (M)}
    const uint64_t globalDimA[] = {(uint64_t)k, (uint64_t)m};
    const uint64_t globalStrideA[] = {sizeof(float), (uint64_t)k * sizeof(float)};
    const cuuint32_t boxDimA[] = {BK, BM};

    CHECK_CUDA_DRIVER(cuTensorMapEncodeTiled(
        &tensor_map_A,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        (void*)d_A,
        globalDimA,
        globalStrideA + 1,
        boxDimA,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    // Tensor B (K x N) -> {inner (N), outer (K)}
    const uint64_t globalDimB[] = {(uint64_t)n, (uint64_t)k};
    const uint64_t globalStrideB[] = {sizeof(float), (uint64_t)n * sizeof(float)};
    const cuuint32_t boxDimB[] = {BN, BK};

    CHECK_CUDA_DRIVER(cuTensorMapEncodeTiled(
        &tensor_map_B,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        (void*)d_B,
        globalDimB,
        globalStrideB + 1,
        boxDimB,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    dim3 blockDim(BM * BN / (TM * TN));
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    size_t shared_mem_size = (BM * BK + BK * BN) * sizeof(float);

    vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim, shared_mem_size, stream>>>(
        tensor_map_A,
        tensor_map_B,
        d_C, m, n, k);
}

int main() {
    CHECK_CUDA_DRIVER(cuInit(0));
    CUcontext ctx;
    CHECK_CUDA_DRIVER(cuCtxGetCurrent(&ctx));

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int TM = 8;
    constexpr int TN = 4;

    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;

    // Set kernel attributes (carveout preference, max dynamic shared memory)
    CHECK_CUDA(cudaFuncSetAttribute(
        (const void*)vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    ));
     size_t shared_mem_size = (BM * BK + BK * BN) * sizeof(float);
     if (shared_mem_size > deviceProp.sharedMemPerBlock) {
         std::cerr << "Warning: Requested shared memory (" << shared_mem_size << ") exceeds device limit (" << deviceProp.sharedMemPerBlock << ")" << std::endl;
     }
     CHECK_CUDA(cudaFuncSetAttribute(
         (const void*)vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>,
         cudaFuncAttributeMaxDynamicSharedMemorySize,
         shared_mem_size
     ));

    // Default matrix dimensions
    int m = 4096;
    int n = 2048;
    int k = 512;

    std::cout << "Running Vectorized 2D block tiling (TMA with TensorMap) matrix multiplication benchmark:" << std::endl;

    float avg_time = run_benchmark<float>(
        launch_vectorized_2d_block_tiling_matmul, m, n, k
    );

    return 0;
}