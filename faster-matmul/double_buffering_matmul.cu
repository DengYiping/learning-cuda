#include "faster_matmul.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>


#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_2d_block_tiling_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    /* ---------------- Shared memory layout (double buffered) -------------
       |  A_0  |  B_0  |  A_1  |  B_1  |
       --------------------------------------------------------------------*/
    extern __shared__ float smem[];
    float* smem_A[2] = { smem,
                         smem + BM * BK + BN * BK };
    float* smem_B[2] = { smem + BM * BK,
                         smem + 2 * (BM * BK) + BN * BK }; 

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


    /* ---------------- Pipeline (double buffering with cuda::memcpy_async) ----------------
       |  A_0  |  B_0  |  A_1  |  B_1  |
       --------------------------------------------------------------------*/
    // Use a 2-stage pipeline (double buffering)
    constexpr auto scope = cuda::thread_scope_thread;
    auto thread = cooperative_groups::this_thread();
    auto group = cooperative_groups::this_thread_block();
    cuda::pipeline<scope> pipe = cuda::make_pipeline();

    // Stage indices (ping-pong)
    int write_stage = 0;  // stage currently being filled
    int read_stage  = 0;  // stage currently being consumed

    // ---------------- Prime the pipeline : load the very first tile ----------------
    pipe.producer_acquire();
    // load A_tile into shared_A
    for (int idx = threadIdx.x; idx < (BM * BK) / 4; idx += blockDim.x) {
        // Each row of the A tile contains BK/4 float4 elements.
        int row   = idx / (BK / 4);     // which row in the tile
        int col4  = idx % (BK / 4);     // which float4 within the row

        cuda::memcpy_async(
            thread,
            &reinterpret_cast<float4*>(smem_A[write_stage])[idx],
            &reinterpret_cast<const float4*>(A)[row * (K / 4) + col4],
            cuda::aligned_size_t<16>(sizeof(float4)),
            pipe);
    }

    // load B_tile into shared_B
    for (int idx = threadIdx.x; idx < (BK * BN) / 4; idx += blockDim.x) {
        // Each row of the B tile contains BN/4 float4 elements.
        int row   = idx / (BN / 4);
        int col4  = idx % (BN / 4);

        cuda::memcpy_async(
            thread,
            &reinterpret_cast<float4*>(smem_B[write_stage])[idx],
            &reinterpret_cast<const float4*>(B)[row * (N / 4) + col4],
            cuda::aligned_size_t<16>(sizeof(float4)),
            pipe);
    }
    pipe.producer_commit();

    // Advance global pointers to the next tile along K
    A += BK;            // next A tile starts BK columns to the right
    B += BK * N;        // next B tile is BK rows down

    // number of K-tiles we will iterate over
    const uint num_tiles = (K + BK - 1) / BK; // assume K % BK == 0, but keep generic

    for (uint tile = 0; tile < num_tiles; ++tile) {
                // ---------------- Preload the next tile (if any) while computation continues ----------------
        if (tile + 1 < num_tiles) {
            write_stage ^= 1;            // toggle between 0 and 1
            pipe.producer_acquire();
            // load A_tile into shared_A
            for (int idx = threadIdx.x; idx < (BM * BK) / 4; idx += blockDim.x) {
                int row  = idx / (BK / 4);
                int col4 = idx % (BK / 4);
                cuda::memcpy_async(
                    thread,
                    &reinterpret_cast<float4*>(smem_A[write_stage])[idx],
                    &reinterpret_cast<const float4*>(A)[row * (K / 4) + col4],
                    cuda::aligned_size_t<16>(sizeof(float4)),
                    pipe);
            }

            // load B_tile into shared_B
            for (int idx = threadIdx.x; idx < (BK * BN) / 4; idx += blockDim.x) {
                int row  = idx / (BN / 4);
                int col4 = idx % (BN / 4);
                cuda::memcpy_async(
                    thread,
                    &reinterpret_cast<float4*>(smem_B[write_stage])[idx],
                    &reinterpret_cast<const float4*>(B)[row * (N / 4) + col4],
                    cuda::aligned_size_t<16>(sizeof(float4)),
                    pipe);
            }
            pipe.producer_commit();

            // Advance global pointers for A and B to point at the subsequent tile
            A += BK;
            B += BK * N;
        }
        // Wait for the tile in "read_stage" to be fully copied to shared memory
        pipe.consumer_wait();
        group.sync();          // ensure coherence before compute

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
        // Toggle read_stage to the buffer we just finished copying
        read_stage ^= 1;
        group.sync();          // Ensure all threads finished computing before releasing stage
        pipe.consumer_release();
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
void launch_vectorized_2d_block_tiling_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int m, int n, int k, cudaStream_t stream) {
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
    
    size_t smem_bytes = 2 * (BM * BK + BN * BK) * sizeof(float);
    vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim, smem_bytes, stream>>>(d_A, d_B, d_C, m, n, k);
}

int main() {
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


    // Set shared memory carveout for this kernel
    CHECK_CUDA(cudaFuncSetAttribute(
        vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100
    ));
    // Set shared memory size for this kernel
    CHECK_CUDA(cudaFuncSetAttribute(
        vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        220 * 1024
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
