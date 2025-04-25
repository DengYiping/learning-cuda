// sync_db_warp_matmul.cu
// -----------------------------------------------------------------------------
// 64 × 128 × 8 tile GEMM with
//   • 128-bit (float4) synchronous vector loads/stores
//   • Transposed A stored in shared memory for bank-conflict-free accesses
//   • Warp-tiling strategy adapted from the reference kernel supplied by user
//   • Two-stage shared-memory double buffering across the K dimension
//   • Pure CUDA C++ (ASCII only, no cp.async, no memcpy_async)
// -----------------------------------------------------------------------------

#include "faster_matmul.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))

// -----------------------------------------------------------------------------
// Compile-time configuration
// -----------------------------------------------------------------------------
constexpr uint BM      = 64;   // Thread-block tile size (rows  of C)
constexpr uint BN      = 128;  // Thread-block tile size (cols  of C)
constexpr uint BK      = 8;    // Thread-block tile size (depth)

constexpr uint WM      = 32;   // Warp tile size in M
constexpr uint WN      = 64;   // Warp tile size in N
constexpr uint WNITER  = 2;    // How many slices a warp tile is broken into along N
constexpr uint TM      = 4;    // Per-thread tile size in M
constexpr uint TN      = 4;    // Per-thread tile size in N

// Derived quantities ----------------------------------------------------------
constexpr uint WARPSZ              = 32U;          // warp size as a constexpr for compile-time calculations
const uint WARPSIZE                = 32U;          // runtime constant as well

constexpr uint NUM_WARPS   = (BM / WM) * (BN / WN);          // 4 warps
constexpr uint NUM_THREADS = NUM_WARPS * WARPSZ;             // 128 threads

// Derived constants used in copy routines
constexpr uint ROWSTRIDE_A = (NUM_THREADS * 4) / BK;
constexpr uint ROWSTRIDE_B = NUM_THREADS / (BN / 4);

constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER); // 2
constexpr uint WSUBM  = WM / WMITER; // 16
constexpr uint WSUBN  = WN / WNITER; // 32

// Size (in floats) of one stage in shared memory
constexpr uint STAGE_SIZE = BK * (BM + BN); // 8 * 192 = 1536 floats

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS)
void sync_db_warp_matmul_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    // Dynamic shared memory organisation
    extern __shared__ float smem[];
    float* As[2];
    float* Bs[2];
    As[0] = smem;
    Bs[0] = As[0] + BK * BM;
    As[1] = Bs[0] + BK * BN;          // start of stage 1
    Bs[1] = As[1] + BK * BM;

    // Block coordinates in output tile grid
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // Warp identification inside the block
    const uint warp_id   = threadIdx.x / WARPSIZE;       // 0..3
    const uint warp_row  = warp_id / (BN / WN);          // 0..1
    const uint warp_col  = warp_id % (BN / WN);          // 0..1

    // Thread lane id inside the warp
    const uint lane_id      = threadIdx.x % WARPSIZE;     // 0..31
    const uint lane_row     = lane_id / (WSUBN / TN);     // placement within WSUBM (0..7)
    const uint lane_col     = lane_id % (WSUBN / TN);     // placement within WSUBN/TN (0..3)

    // Global pointers adjusted to the beginning of this C thread-block tile
    const float* A_block = A + block_row * BM * K;
    const float* B_block = B + block_col * BN;
    float*       C_block = C + block_row * BM * N + block_col * BN;

    // ---------------------------------------------------------------------
    // Registers
    // ---------------------------------------------------------------------
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f}; // 64 floats
    float regM[WMITER * TM] = {0.0f}; // 8 floats
    float regN[WNITER * TN] = {0.0f}; // 8 floats

    // Helper indices for SMEM load tiling
    const uint innerRowA = threadIdx.x / (BK / 4);   // 0..NUM_THREADS*4/BK-1
    const uint innerColA = threadIdx.x % (BK / 4);   // 0..(BK/4)-1
    constexpr uint rowStrideA = ROWSTRIDE_A;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = ROWSTRIDE_B;

    // Double-buffer control
    uint read_stage  = 0;
    uint write_stage = 1;

    // ---------------------------------------------------------------------
    // Pre-load tile 0 into shared memory (stage 0)
    // ---------------------------------------------------------------------
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const float4* src_ptr = reinterpret_cast<const float4*>(
            &A_block[(innerRowA + offset) * K + innerColA * 4]);
        float4 vec = *src_ptr; // 128-bit load

        // Transpose while storing: shared layout (BK×BM)
        float* As_ptr = As[read_stage] + (innerColA * 4 + 0) * BM + innerRowA + offset;
        As_ptr[0]                = vec.x;
        As_ptr[BM]               = vec.y; // (col+1) row offset by BM
        As_ptr[BM * 2]           = vec.z;
        As_ptr[BM * 3]           = vec.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        float4* dst_ptr = reinterpret_cast<float4*>(
            &Bs[read_stage][(innerRowB + offset) * BN + innerColB * 4]);
        const float4* src_ptr = reinterpret_cast<const float4*>(
            &B_block[(innerRowB + offset) * N + innerColB * 4]);
        *dst_ptr = *src_ptr; // 128-bit copy
    }
    __syncthreads(); // Tile 0 ready

    // ---------------------------------------------------------------------
    // Loop over K tiles
    // ---------------------------------------------------------------------
    for (int k_base = 0; k_base < K; k_base += BK) {
        // -------------------------------------------------------------
        // If there is a next tile, start loading it into write_stage
        // BEFORE we begin compute on read_stage.  Because loads are
        // synchronous we cannot overlap computation, but using two
        // buffers avoids read/write hazards.
        // -------------------------------------------------------------
        if (k_base + BK < K) {
            // Global pointers for the upcoming tile
            const float* A_next = A_block + (k_base + BK);
            const float* B_next = B_block + (k_base + BK) * N;

            // Load A_next into As[write_stage] (transpose on the fly)
            for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
                const float4* src_ptr = reinterpret_cast<const float4*>(
                    &A_next[(innerRowA + offset) * K + innerColA * 4]);
                float4 vec = *src_ptr;

                float* As_ptr = As[write_stage] + (innerColA * 4) * BM + innerRowA + offset;
                As_ptr[0]                = vec.x;
                As_ptr[BM]               = vec.y;
                As_ptr[BM * 2]           = vec.z;
                As_ptr[BM * 3]           = vec.w;
            }

            // Load B_next into Bs[write_stage]
            for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
                float4* dst_ptr = reinterpret_cast<float4*>(
                    &Bs[write_stage][(innerRowB + offset) * BN + innerColB * 4]);
                const float4* src_ptr = reinterpret_cast<const float4*>(
                    &B_next[(innerRowB + offset) * N + innerColB * 4]);
                *dst_ptr = *src_ptr;
            }
        }

        __syncthreads(); // Ensure data in read_stage is stable before compute

        // -------------------------------------------------------------
        // Compute C sub-tile for this K tile
        // -------------------------------------------------------------
        float* As_tile = As[read_stage]; // (BK × BM)
        float* Bs_tile = Bs[read_stage]; // (BK × BN)

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load a strip of A into registers
            for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
                for (uint i = 0; i < TM; ++i) {
                    regM[wSubRow * TM + i] =
                        As_tile[dotIdx * BM + warp_row * WM + wSubRow * WSUBM +
                                 lane_row * TM + i];
                }
            }

            // Load a strip of B into registers
            for (uint wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
                for (uint i = 0; i < TN; ++i) {
                    regN[wSubCol * TN + i] =
                        Bs_tile[dotIdx * BN + warp_col * WN + wSubCol * WSUBN +
                                lane_col * TN + i];
                }
            }

            // Multiply-accumulate into threadResults
            for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
                for (uint wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
                    for (uint m = 0; m < TM; ++m) {
                        for (uint n = 0; n < TN; ++n) {
                            threadResults[(wSubRow * TM + m) * (WNITER * TN) +
                                          wSubCol * TN + n] +=
                                regM[wSubRow * TM + m] * regN[wSubCol * TN + n];
                        }
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // Switch stages for next iteration
        // -------------------------------------------------------------
        read_stage  ^= 1;
        write_stage ^= 1;
        __syncthreads(); // Ensure loads into write_stage finished before reuse
    }

    // ---------------------------------------------------------------------
    // Write accumulated results to global memory (vectorised stores)
    // ---------------------------------------------------------------------
    for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
        for (uint wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
            float* C_tile = C_block + (warp_row * WM + wSubRow * WSUBM) * N +
                            warp_col * WN + wSubCol * WSUBN;

            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; n += 4) {
                    float4* C_vec = reinterpret_cast<float4*>(
                        &C_tile[(lane_row * TM + m) * N + lane_col * TN + n]);
                    uint resBase = (wSubRow * TM + m) * (WNITER * TN) +
                                   wSubCol * TN + n;

                    float4 out_vec;
                    out_vec.x = threadResults[resBase + 0];
                    out_vec.y = threadResults[resBase + 1];
                    out_vec.z = threadResults[resBase + 2];
                    out_vec.w = threadResults[resBase + 3];

                    *C_vec = out_vec;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Launcher
// -----------------------------------------------------------------------------
void launch_sync_db_warp_matmul(const float* d_A,
                                const float* d_B,
                                float* d_C,
                                int m, int n, int k,
                                cudaStream_t stream) {

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    size_t smem_bytes = 2 * STAGE_SIZE * sizeof(float); // double buffer

    // Set dynamic shared memory limit and carve-out preference
    CHECK_CUDA(cudaFuncSetAttribute(
        sync_db_warp_matmul_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    CHECK_CUDA(cudaFuncSetAttribute(
        sync_db_warp_matmul_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    sync_db_warp_matmul_kernel<<<gridDim, blockDim, smem_bytes, stream>>>(
        d_A, d_B, d_C, m, n, k);
}

#ifndef UNIT_TEST
int main() {
    int m = 4096;
    int n = 2048;
    int k = 512;

    std::cout << "Running sync_db_warp_matmul benchmark:" << std::endl;
    run_benchmark<float>(launch_sync_db_warp_matmul, m, n, k);
    return 0;
}
#endif
