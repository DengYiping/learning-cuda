#include "faster_matmul.cuh"
#include <cuda/pipeline>  // still required for cuda::thread_scope_thread if used elsewhere (kept for completeness)
#include <cooperative_groups.h>

/*
   This kernel is identical to async_global_to_shared_matmul.cu except that the
   asynchronous copy from global to shared memory is implemented via the new
   `cp.async.bulk` PTX instruction that became available with the Hopper
   architecture (SM 90 / H100).  The variant used below performs a **strided**
   bulk copy so that we can transfer entire *rows* of the current K‑tile in
   one instruction instead of issuing many individual 16‑byte transactions as
   done with `cuda::memcpy_async`.

   For architectures that do **not** support `cp.async.bulk` we automatically
   fall back to the original `cuda::memcpy_async` implementation so that this
   file still compiles and runs on earlier GPUs (Ampere, Ada, …).

   NOTE: The PTX syntax for `cp.async.bulk` is subject to change – the inline
   assembly below follows the syntax that shipped with CUDA 12.2 / PTX 8.7.
*/

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

/* ------------------------------- Helpers ----------------------------------- */

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// The Hopper asynchronous bulk copy instructions rely on a *memory barrier*
// object that lives in shared memory.  Each outstanding bulk copy is tagged
// with such a barrier so that the producer can signal completion and the
// consumer can wait for it.  We expose a minimal C++ wrapper that hides the
// low‑level PTX details and models the same "acquire/commit / wait / release"
// semantics that `cuda::pipeline` offered for the previous `cp.async` API.

// A single 64‑bit slot in shared memory is sufficient for one barrier.
struct BulkCopyBarrier {
    // Pointer into shared memory returned by mbarrier.init
    unsigned long long *ptr;

    __device__ void init(unsigned long long *storage, unsigned expected_count) {
        ptr = storage;
        // `mbarrier.init.shared::cta`   BARRIER, EXPECTED, 0;
        asm volatile ("mbarrier.init.shared::cta.b64 %0, %1, 0;\n"
                      : /* no outputs */
                      : "l"(ptr), "r"(expected_count));
    }

    // Tell the barrier that one asynchronous transaction has been launched.
    __device__ void arrive() {
        asm volatile ("mbarrier.arrive.expect_tx.b64 %0;\n" :: "l"(ptr));
    }

    // Wait until *all* transactions that arrived at this barrier have
    // completed (i.e. reached shared memory).
    __device__ void wait() {
        asm volatile ("mbarrier.try_wait.parity.b64 %0;\n" :: "l"(ptr));
    }
};

// Issue a strided cp.async.bulk from global to shared memory.  Each call will
// copy   rows * bytes_per_row   bytes in total.  Source rows are separated by
// `src_stride_bytes` in global memory;  destination rows are contiguous in
// shared memory.
__device__ inline void cp_async_bulk_strided_shared_global(
        void            *dst,                 // base in shared memory
        const void      *src,                 // base in global memory
        unsigned         bytes_per_row,       // typically BK*sizeof(float)
        unsigned         rows,                // rows to copy per instruction
        unsigned         src_stride_bytes,    // leading dimension of A/B in bytes
        BulkCopyBarrier &bar) {

    // Hopper PTX allows copying up to 16 KiB per instruction.  For our use
    // case we stay well below that limit (e.g. 128 rows × 256 B = 32 KiB), but
    // we conservatively restrict ourselves to one row per instruction to keep
    // the implementation simple and avoid register pressure.

    for (unsigned r = 0; r < rows; ++r) {
        const void *src_row = static_cast<const char*>(src)  + r * src_stride_bytes;
        void       *dst_row = static_cast<char*>(dst)        + r * bytes_per_row;

        // cp.async.bulk.shared.global.mbarrier::cta [dst], [src], bytes, [barrier];
        asm volatile (
            "cp.async.bulk.shared.global.mbarrier::cta [%0], [%1], %2, %3;\n"
            : /* no outputs */
            : "r"(dst_row), "l"(src_row), "n"(bytes_per_row), "r"(bar.ptr));

        // Inform the barrier that we issued one TX so that later waits know
        // how many transfers to expect.
        bar.arrive();
    }
}

#endif  // __CUDA_ARCH__ >= 900


/* -------------------------  MatMul Kernel  --------------------------------- */

template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_2d_block_tiling_matmul(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float*       __restrict__ C,
        const int M, const int N, const int K) {

    extern __shared__ float shared_A[];
    float* shared_B = shared_A + BM * BK;

    // blockIdx.x : block index in N dim  (columns)
    // blockIdx.y : block index in M dim  (rows)

    // Each thread computes a TM×TN micro‑tile of C.
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Move base pointers so that A/B point at the first K‑tile of the block
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float thread_results[TM][TN] = {0.0f};
    float reg_M[TM];
    float reg_N[TN];

    /* ----------------------------------------------------------------------
       Asynchronous copy helpers
    ---------------------------------------------------------------------- */

    auto  block  = cooperative_groups::this_thread_block();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // One shared memory barrier is sufficient because we keep a single
    // in‑flight tile (no double buffering in this kernel).
    __shared__ unsigned long long barrier_storage;
    BulkCopyBarrier mbar;
    if (threadIdx.x == 0) {
        // Expect BM (rows of A) + BK (rows of B) bulk‑copy operations.
        mbar.init(&barrier_storage, /*expected_count=*/BM + BK);
    }
    __syncthreads();
#endif

    // Assume K is divisible by BK.  Outer loop iterates over K‑tiles.
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        /* --------------------------------------------------------------
           Load A/B tile using cp.async.bulk with a *strided* source.  We
           copy one row per thread so that we transfer an entire BK‑element
           row (BK * 4 bytes) in a single instruction.
        -------------------------------------------------------------- */

        // Copy tile of A (BM×BK) from global to shared.
        for (int row = threadIdx.x; row < BM; row += blockDim.x) {
            const float *src = A + row * K;        // row pointer in A
            float       *dst = shared_A + row * BK; // row pointer in smem
            cp_async_bulk_strided_shared_global(/*dst*/ dst,
                                              /*src*/ src,
                                              /*bytes_per_row*/ BK * sizeof(float),
                                              /*rows*/ 1,
                                              /*src_stride_bytes*/ K * sizeof(float),
                                              /*barrier*/ mbar);
        }

        // Copy tile of B (BK×BN) – here rows are **columns** of C.
        for (int row = threadIdx.x; row < BK; row += blockDim.x) {
            const float *src = B + row * N;        // row pointer in B (length BN)
            float       *dst = shared_B + row * BN;
            cp_async_bulk_strided_shared_global(dst, src,
                                              /*bytes_per_row*/ BN * sizeof(float),
                                              1,
                                              N * sizeof(float),
                                              mbar);
        }

        // Wait until both copies reached shared memory
        mbar.wait();

        block.sync();

        /* ------------------------------------------------------------------
           Tile is now resident in shared memory → compute partial C.
        ------------------------------------------------------------------ */

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
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

        // Advance to next K‑tile in global memory
        A += BK;
        B += BK * N;
    }

    // Store results
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; j += 4) {
            reinterpret_cast<float4*>(&C[(thread_row * TM + i) * N + (thread_col * TN + j)])[0] =
                reinterpret_cast<float4*>(&thread_results[i][j])[0];
        }
    }
}


// Launcher – identical to async_global_to_shared_matmul.cu -------------------

void launch_vectorized_2d_block_tiling_matmul(const float* __restrict__ d_A,
                                              const float* __restrict__ d_B,
                                              float*       __restrict__ d_C,
                                              int m, int n, int k,
                                              cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int TM = 8;
    constexpr int TN = 4;

    dim3 blockDim(BM * BN / (TM * TN));
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    vectorized_2d_block_tiling_matmul<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim, (BM + BN) * BK * sizeof(float), stream>>>(d_A, d_B, d_C, m, n, k);
}


// Simple standalone test driver (same as original file) ----------------------

int main() {
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Device name: " << deviceProp.name << std::endl;

    // Matrix dimensions
    int m = 4096, n = 2048, k = 512;

    std::cout << "Running cp.async.bulk‑based matrix multiplication benchmark:" << std::endl;

    float avg_time = run_benchmark<float>(launch_vectorized_2d_block_tiling_matmul, m, n, k);

    std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;

    return 0;
}
