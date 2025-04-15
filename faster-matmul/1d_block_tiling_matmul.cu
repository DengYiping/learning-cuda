#include "faster_matmul.cuh"

#define BLOCK_SIZE 32

__global__ void sram_matmul(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // threadIdx.x changes in the same warp
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;


    if (i < M && j < N) {
        // sliding window of BLOCK_SIZE x BLOCK_SIZE
        // sliding on the K dimension
        float sum = 0.0f;
        for (int w_idx = 0; w_idx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; w_idx++) {
            int k_start = w_idx * BLOCK_SIZE;

            // init shared_A, shared_B and shared_C
            // Check bounds for shared_A and shared_B, add assign value to 0 if out of bounds
            if (k_start + threadIdx.x < K) {
                shared_A[threadIdx.y][threadIdx.x] = A[i * K + k_start + threadIdx.x];
            }

            if (k_start + threadIdx.y < K) {
                shared_B[threadIdx.y][threadIdx.x] = B[(k_start + threadIdx.y) * N + j];
            }

            __syncthreads(); // wait for all threads to finish loading data
            for (int k = 0; k < BLOCK_SIZE && (k_start + k) < K; k++) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
            __syncthreads(); // avoid over-writing shared_A and shared_B before all threads finish the computation
        }
        C[i * N + j] = sum;
    }
}

// Kernel launcher function
void launch_sram_matmul(float* d_A, float* d_B, float* d_C, int m, int n, int k, cudaStream_t stream) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // gridDim is reversed because we want to access the memory with changing j first instead of 
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    
    sram_matmul<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, m, n, k);
}

int main() {
    // Default matrix dimensions
    int m = 4096; // Matrix A: m x k
    int n = 2048; // Matrix B: k x n, Matrix C: m x n
    int k = 512;
    
    std::cout << "Running sram bank conflict free matrix multiplication benchmark:" << std::endl;
    
    // Run the benchmark with the naive matrix multiplication kernel
    float avg_time = run_benchmark<float>(
        launch_sram_matmul, m, n, k
    );
    
    return 0;
}

