#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublasXt.h>

#define M 8192
#define N 4096
#define K 2048

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define PRINT_MATRIX(mat, rows, cols) \
    printf("Top-left 3x3 corner:\n"); \
    for (int i = 0; i < 3 && i < rows; i++) { \
        for (int j = 0; j < 3 && j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

void cpu_matmul(float *A, float *B, float *C) {
    // For large matrices, only compute the top-left 3x3 corner for verification
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * 3 + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
  
    // Allocate host memory for matrices using pinned memory
    float *A, *B, *h_cpu, *h_cublasxt;
  
    CHECK_CUDA(cudaMallocHost(&A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_cpu, 9 * sizeof(float)));  // Only need 3x3 for CPU verification
    CHECK_CUDA(cudaMallocHost(&h_cublasxt, M * N * sizeof(float)));

    if (!A || !B || !h_cpu || !h_cublasxt) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    // CPU matmul for verification (only computes top-left 3x3)
    cpu_matmul(A, B, h_cpu);

    // cublasXt setup
    cublasXtHandle_t handle;
    CHECK_CUBLAS(cublasXtCreate(&handle));

    // Get the number of GPU devices available
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    printf("Number of GPU devices available: %d\n", deviceCount);

    // Set the devices to use
    int devices[deviceCount];
    for (int i = 0; i < deviceCount; i++) {
        devices[i] = i;
    }
    CHECK_CUBLAS(cublasXtDeviceSelect(handle, deviceCount, devices));

    // Set block size
    CHECK_CUBLAS(cublasXtSetBlockDim(handle, 1024));  // Typical block size, can be tuned

    // Set cublasXt GEMM parameters
    float alpha = 1.0f, beta = 0.0f;
    
    // Perform SGEMM: C = alpha*A*B + beta*C
    // Note: cublasXt uses column-major format, but we have row-major data
    // So we compute B^T * A^T = (A*B)^T
    CHECK_CUBLAS(cublasXtSgemm(
        handle,
        CUBLAS_OP_N,       // No transpose for B^T
        CUBLAS_OP_N,       // No transpose for A^T
        N, M, K,           // Dimensions: N x M = K x M * N x K (for result C)
        &alpha,            // Alpha coefficient
        B, N,              // B matrix (treated as B^T in column-major)
        A, K,              // A matrix (treated as A^T in column-major)
        &beta,             // Beta coefficient
        h_cublasxt, N      // Output C matrix (column-major)
    ));

    // Synchronize to ensure computation is complete
    CHECK_CUDA(cudaDeviceSynchronize());

    // Print results for verification
    printf("CPU result (top-left 3x3 only):\n");
    PRINT_MATRIX(h_cpu, 3, 3);
    printf("cublasXt SGEMM result:\n");
    PRINT_MATRIX(h_cublasxt, M, N);

    // Free resources
    CHECK_CUBLAS(cublasXtDestroy(handle));
    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
    CHECK_CUDA(cudaFreeHost(h_cpu));
    CHECK_CUDA(cudaFreeHost(h_cublasxt));

    return 0;
} 