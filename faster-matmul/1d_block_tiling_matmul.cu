#include "faster_matmul.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <int BM, int BN, int BK, int TM>
__global__ void block_tiling_matmul_1d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];

    // blockIdx.x is the block id in the N dimension, aka the column index of the block
    // blockIdx.y is the block id in the M dimension, aka the row index of the block

    // Each warp will calculate 32 * TM elements, with 32 being the columnar dim.
    // Num threads = BM * BN / TM, we will 1d tiling on the M dimension.
    const uint thread_col = threadIdx.x % BN;
    const uint thread_row = threadIdx.x / BN;

    // Move blocktile to beginning of A's row and B's column
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint inner_col_a = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint inner_row_a = threadIdx.x / BK;
    const uint inner_col_b = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint inner_row_b = threadIdx.x / BN;

    float thread_results[TM] = {0.0f};

    // Assume K is divisible by BK
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A and B tiles into shared memory j
        shared_A[inner_row_a][inner_col_a] = A[inner_row_a * K + inner_col_a];
        shared_B[inner_row_b][inner_col_b] = B[inner_row_b * N + inner_col_b];

        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // Perform matrix multiplication
        for (uint dot_idx = 0; dot_idx < BK; dot_idx++) {
            // This is cached & reused for each thread in the warp
            float tmp = shared_B[dot_idx][thread_col];

            // We are reading TM elemets from A[thread_row * TM : thread_row * TM + TM][dot_idx]
            // and multiply with the cached B[thread_col][dot_idx]
            #pragma unroll
            for (uint res_idx = 0; res_idx < TM; res_idx++) {
                thread_results[res_idx] += shared_A[thread_row * TM + res_idx][dot_idx] * tmp;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint res_idx = 0; res_idx < TM; res_idx++) {
        C[(thread_row * TM + res_idx) * N + thread_col] = thread_results[res_idx];
    }
}

// Kernel launcher function
void launch_1d_block_tiling_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int m, int n, int k, cudaStream_t stream) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    dim3 blockDim(BM * BN / TM);
    // Reversing order to optimize L2 cache access. Grid will move on the N dimension fast and M dimension slow.
    // With row-major layout, this is more cache-friendly.
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    
    block_tiling_matmul_1d<BM, BN, BK, TM><<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, m, n, k);
}

int main() {
    // Default matrix dimensions
    int m = 4096; // Matrix A: m x k
    int n = 2048; // Matrix B: k x n, Matrix C: m x n
    int k = 512;
    
    std::cout << "Running 1d_block_tiling_matmul benchmark:" << std::endl;
    
    // Run the benchmark with the naive matrix multiplication kernel
    float avg_time = run_benchmark<float>(
        launch_1d_block_tiling_matmul, m, n, k
    );
    
    return 0;
}
