#include "faster_matmul.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void block_tiling_matmul_2d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];

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

    const uint inner_col_a = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint inner_row_a = threadIdx.x / BK;
    const uint inner_col_b = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint inner_row_b = threadIdx.x / BN;

    float thread_results[TM][TN] = {0.0f};
    float reg_M[TM];
    float reg_N[TN];

    const uint stride_A = blockDim.x / BK;
    const uint stride_B = blockDim.x / BN;

    // Assume K is divisible by BK. Outer loop is over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A and B tiles into shared memory

        // For shared_A, we need to load BM * BK in total, and we've got (BM * BN) / (TM * TN) threads
        // So each thread needs to load (BM * BK) / ((BM * BN) / (TM * TN)) = BK * TM * TN / BN
        // This is equivalent to traversing on the BM dimension with stride_A = BM / (BK * TM * TN / BN) = ((BM * BN) / (TM * TN)) / BK =
        // blockDim.x / BK
        #pragma unroll
        for (int j = 0; j < BM; j += blockDim.x / BK) {
            shared_A[inner_row_a + j][inner_col_a] = A[(inner_row_a + j) * K + inner_col_a];
        }
        // For shared_B, we need to load BK * BN in total, and we've got (BM * BN) / (TM * TN) threads
        // So each thread needs to load (BK * BN) / ((BM * BN) / (TM * TN)) = BK * TM * TN / BM
        // This is equivalent to traversing on the BK dimension with stride_B = BK / (BK * TM * TN / BM) = ((BM * BN) / (TM * TN)) / BN =
        // blockDim.x / BN
        #pragma unroll
        for (int j = 0; j < BK; j += stride_B) {
            shared_B[inner_row_b + j][inner_col_b] = B[(inner_row_b + j) * N + inner_col_b];
        }
        __syncthreads();
        // advance blocktile
        A += BK;
        B += BK * N;

        // Perform matrix multiplication
        #pragma unroll
        for (uint dot_idx = 0; dot_idx < BK; dot_idx++) {
            // Outer dot product over reg_M and reg_N
            #pragma unroll
            for (uint i = 0; i < TM; i++) {
                reg_M[i] = shared_A[thread_row * TM + i][dot_idx];
            }

            #pragma unroll
            for (uint j = 0; j < TN; j++) {
                reg_N[j] = shared_B[dot_idx][thread_col * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    thread_results[i][j] += reg_M[i] * reg_N[j];
                }
            }
        }

        __syncthreads();
    }

    // Store the results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            C[(thread_row * TM + i) * N + (thread_col * TN + j)] = thread_results[i][j];
        }
    }
}

// Kernel launcher function
void launch_2d_block_tiling_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int m, int n, int k, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;

    // Each thread will calculate TM * TN elements
    dim3 blockDim(BM * BN / (TM * TN)); // 256 threads
    // Reversing order to optimize L2 cache access. Grid will move on the N dimension fast and M dimension slow.
    // With row-major layout, this is more cache-friendly.
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    
    block_tiling_matmul_2d<BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, m, n, k);
}

int main() {
    // Default matrix dimensions
    int m = 4096; // Matrix A: m x k
    int n = 2048; // Matrix B: k x n, Matrix C: m x n
    int k = 512;
    
    std::cout << "Running 2D block tiling matrix multiplication benchmark:" << std::endl;
    
    // Run the benchmark with the naive matrix multiplication kernel
    float avg_time = run_benchmark<float>(
        launch_2d_block_tiling_matmul, m, n, k
    );
    
    return 0;
}
