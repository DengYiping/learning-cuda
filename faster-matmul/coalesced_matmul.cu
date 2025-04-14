#include "faster_matmul.cuh"

#define BLOCK_SIZE 32

__global__ void coalesced_matmul(float* A, float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;


    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

// Kernel launcher function
void launch_coalesced_matmul(float* d_A, float* d_B, float* d_C, int m, int n, int k, cudaStream_t stream) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // gridDim is reversed because we want to access the memory with changing j first instead of 
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    
    coalesced_matmul<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, m, n, k);
}

int main() {
    // Default matrix dimensions
    int m = 4096; // Matrix A: m x k
    int n = 2048; // Matrix B: k x n, Matrix C: m x n
    int k = 512;
    
    std::cout << "Running coalesced matrix multiplication benchmark:" << std::endl;
    
    // Run the benchmark with the naive matrix multiplication kernel
    float avg_time = run_benchmark<float>(
        launch_coalesced_matmul, m, n, k
    );
    
    return 0;
}

