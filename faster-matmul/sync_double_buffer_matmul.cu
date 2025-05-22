// sync_double_buffer_matmul.cu
// ---------------------------------------------------------------
// Synchronous, 128-bit vectorised, double-buffered matrix multiply
// C = A * B
//
// Requirements (from user request):
//   • Use 128-bit (float4) *synchronised* loads – no cp.async or memcpy_async.
//   • Double buffer in shared memory (two SRAM tiles) plus register staging.
//   • Transpose matrix A while writing it to shared memory to enable
//     coalesced, bank-conflict-free access during the compute phase.
//   • Keep everything in plain CUDA C++, ASCII-only.
// ---------------------------------------------------------------

#include "faster_matmul.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))

// Tunable parameters (match style of existing kernels)
constexpr int BM = 128;   // rows per block tile of C
constexpr int BN = 256;   // cols per block tile of C
constexpr int BK = 64;    // K-dimension tile depth

constexpr int TM = 8;     // per-thread rows of C
constexpr int TN = 8;     // per-thread cols of C

constexpr int THREADS_PER_BLOCK = (BM / TM) * (BN / TN);

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
__global__ void sync_double_buffer_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
    extern __shared__ float smem[];

    // Shared memory is divided into two stages to implement double buffering
    // Layout (sizes in scalars):
    //   A stage : BK * BM   (A is stored transposed => shape (BK,BM))
    //   B stage : BK * BN
    // Total per stage        = BK * (BM + BN)
    // Two stages             = 2 * BK * (BM + BN)
    
    float* smem_A[2] = { smem,
                         smem + BK * (BM + BN) };
    float* smem_B[2] = { smem + BK * BM,
                         smem + BK * (BM + BN) + BK * BM };

    const int thread_id = threadIdx.x; // 0 .. THREADS_PER_BLOCK-1

    // Which output tile this block is responsible for
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    // Pointers to the beginning of each block tile in global memory
    const float* A_block_ptr = A + block_m * BM * K;
    const float* B_block_ptr = B + block_n * BN;
    float*       C_block_ptr = C + block_m * BM * N + block_n * BN;

    // Coordinates of this thread within the logical thread tile grid
    const int threads_per_row_of_tiles = BN / TN; // e.g. 256/8 = 32
    const int tile_row = thread_id / threads_per_row_of_tiles; // 0 .. BM/TM-1
    const int tile_col = thread_id % threads_per_row_of_tiles; // 0 .. BN/TN-1

    // Accumulator registers
    float accum[TM][TN] = {0.0f};

    // Register buffers for one "k" slice
    float reg_A[TM];
    float reg_B[TN];

    const int num_k_tiles = CEIL_DIV(K, BK);

    auto copy_tile_to_shared = [&](int k_tile_idx, int stage) {
        // ---------------------------------------------------------
        // Copy A tile (BM x BK) from global and store transposed as
        // (BK x BM) in shared memory (stage buffer).
        // ---------------------------------------------------------
        const int vec4_per_A_tile = (BM * BK) / 4; // scalar count divided by 4
        for (int idx = thread_id; idx < vec4_per_A_tile; idx += THREADS_PER_BLOCK) {
            // Original coordinates in A (row, col)
            int scalar_index = idx * 4;
            int row = scalar_index / BK;  // 0 .. BM-1 (row of A)
            int col = scalar_index % BK;  // 0 .. BK-1 (col of A)

            // Pointer to 128-bit chunk in global A (row major)
            const float4* g_ptr = reinterpret_cast<const float4*>(
                A_block_ptr + row * K + k_tile_idx * BK + col);

            float4 vec = *g_ptr; // synchronised 128-bit load

            // Write transposed to shared: (BK rows, BM cols)
                // Global A tile coordinates: row, col (where col is base of float4 from A tile)
                // Shared smem_A coordinates: k_idx (0..BK-1, from col in A), m_idx (0..BM-1, from row in A)
                // vec.x = A[row, col+0] should go to smem_A[col+0, row]
                // vec.y = A[row, col+1] should go to smem_A[col+1, row]
                // vec.z = A[row, col+2] should go to smem_A[col+2, row]
                // vec.w = A[row, col+3] should go to smem_A[col+3, row]
                
                smem_A[stage][(col + 0) * BM + row] = vec.x;
                smem_A[stage][(col + 1) * BM + row] = vec.y;
                smem_A[stage][(col + 2) * BM + row] = vec.z;
                smem_A[stage][(col + 3) * BM + row] = vec.w;
        }

        // ---------------------------------------------------------
        // Copy B tile (BK x BN) without transpose.
        // ---------------------------------------------------------
        const int vec4_per_B_tile = (BK * BN) / 4;
        for (int idx = thread_id; idx < vec4_per_B_tile; idx += THREADS_PER_BLOCK) {
            int scalar_index = idx * 4;
            int row = scalar_index / BN; // 0 .. BK-1
            int col = scalar_index % BN; // 0 .. BN-1

            const float4* g_ptr = reinterpret_cast<const float4*>(
                B_block_ptr + (k_tile_idx * BK + row) * N + col);

            float4 vec = *g_ptr; // 128-bit load

            float4* s_ptr = reinterpret_cast<float4*>(
                smem_B[stage] + row * BN + col);

            *s_ptr = vec; // 128-bit store (aligned because BN multiple of 4)
        }
    };

    // -----------------------------------------------------------------
    // Tile 0 load (stage 0)
    // -----------------------------------------------------------------
    copy_tile_to_shared(0, 0);
    __syncthreads(); // make tile 0 visible to all threads

    int read_stage = 0;
    int write_stage = 1;

    for (int tile_idx = 0; tile_idx < num_k_tiles; ++tile_idx) {
        // -----------------------------------------------
        // Compute using read_stage
        // -----------------------------------------------
        const float* tile_A = smem_A[read_stage]; // (BK x BM) – already transposed
        const float* tile_B = smem_B[read_stage]; // (BK x BN)

        for (int k_inner = 0; k_inner < BK; ++k_inner) {
            // Load a row of A (contiguous after transpose)
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = tile_A[k_inner * BM + (tile_row * TM + i)];
            }

            // Load a segment of B (vectorised)
            for (int j = 0; j < TN; j += 4) {
                reinterpret_cast<float4*>(&reg_B[j])[0] =
                    reinterpret_cast<const float4*>(&tile_B[k_inner * BN + (tile_col * TN + j)])[0];
            }

            // FMA accumulate
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        // -----------------------------------------------
        // Prepare next tile (if any) into write_stage
        // -----------------------------------------------
        if (tile_idx + 1 < num_k_tiles) {
            __syncthreads();              // ensure compute is done before overwrite
            copy_tile_to_shared(tile_idx + 1, write_stage);
            __syncthreads();              // tile ready for next iteration
            read_stage ^= 1;
            write_stage ^= 1;
        }
    }

    // -----------------------------------------------------------------
    // Store the accumulator to global memory (vectorised where possible)
    // -----------------------------------------------------------------
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            reinterpret_cast<float4*>(
                &C_block_ptr[(tile_row * TM + i) * N + (tile_col * TN + j)])[0] =
                reinterpret_cast<float4*>(&accum[i][j])[0];
        }
    }
}

// -----------------------------------------------------------------------------
// Launcher
// -----------------------------------------------------------------------------
void launch_sync_double_buffer_matmul(
    const float* __restrict__ d_A,
    const float* __restrict__ d_B,
    float* __restrict__ d_C,
    int m, int n, int k,
    cudaStream_t stream)
{
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    size_t smem_bytes = 2 * BK * (BM + BN) * sizeof(float);

    // Request the necessary shared memory size for this kernel.
    // This must be done *before* the first launch, otherwise the launch will
    // fail with invalid-argument if the requested dynamic shared memory
    // exceeds the current limit (default 48 KiB).
    CHECK_CUDA(cudaFuncSetAttribute(
        sync_double_buffer_matmul_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    // Ask the driver to prefer allocating shared memory over L1.
    CHECK_CUDA(cudaFuncSetAttribute(
        sync_double_buffer_matmul_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    sync_double_buffer_matmul_kernel<<<gridDim, blockDim, smem_bytes, stream>>>(
        d_A, d_B, d_C, m, n, k);
}

#ifndef UNIT_TEST
int main() {
    int m = 4096;
    int n = 2048;
    int k = 512;

    std::cout << "Running sync_double_buffer_matmul benchmark:" << std::endl;
    run_benchmark<float>(launch_sync_double_buffer_matmul, m, n, k);
    return 0;
}
#endif
