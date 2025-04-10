#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h> // Add this for memset
#include <time.h> // For benchmarking
#include <nvtx3/nvToolsExt.h> // Used for profiling

#define M 1024 // Num of rows in A and C
#define K 512 // Num of columns in A and rows in B
#define N 2048 // Num of columns in B and C

using namespace std;
// Example:
// A:
// [
//  [1, 2, 3],
//  [4, 5, 6]
// ]
//
// B:
// [
//  [1, 2],
//  [3, 4],
//  [5, 6]
// ]
//
// C:
// [
//  [22, 28],
//  [49, 64]
// ]

void h_matmul(float* h_A, float* h_B, float* h_C, int m, int n, int kk) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
      for (int k = 0; k < kk; k++) {
        sum += h_A[i * kk + k] * h_B[k * n + j];
      }
      h_C[i * n + j] = sum;
    }
  }
}

void h_print_matrix(float* matrix, int m, int n) {
  cout << fixed;
  cout << setprecision(3);

  cout << "[" << endl;
  for (int i = 0; i < m; i++) {
    cout << " [";
    for (int j = 0; j < n; j++) {
      cout << matrix[i * n + j] << " ";
    }
    cout << "]" << endl;
  }
  cout << "]" << endl;
}

__global__ void d_matmul(float* d_A, float* d_B, float* d_C, int m, int n, int kk) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m && j < n) {
    float sum = 0.0;
    for (int k = 0; k < kk; k++) {
      sum += d_A[i * kk + k] * d_B[k * n + j];
    }
    d_C[i * n + j] = sum;
  }
}

void h_init_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = (float) rand() / RAND_MAX;
  }
}

void verify_eq(float* A, float* B, int size) {
  for (int i = 0; i < size; i++) {
    if ((A[i] - B[i]) > 1e-3 || (A[i] - B[i]) < -1e-3) {
      cout << "Error at index " << i << ": " << A[i] << " != " << B[i] << endl;
      exit(1);
    }
  }
}

int main(int argc, char** argv) {
  nvtxRangePush("Matmul");
  // Allocate host memory
  float* h_A = (float*) malloc(sizeof(float) * M * K);
  float* h_B = (float*) malloc(sizeof(float) * K * N);
  float* h_C = (float*) malloc(sizeof(float) * M * N);
  float* h_result = (float*) malloc(sizeof(float) * M * N);

  // Initialize matrices
  h_init_matrix(h_A, M, K);
  h_init_matrix(h_B, K, N);
  memset(h_C, 0, sizeof(float) * M * N);

  // Calculate CPU result first
  clock_t cpu_start = clock();
  h_matmul(h_A, h_B, h_C, M, N, K);
  clock_t cpu_end = clock();
  double cpu_elapsed_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
  cout << "Matrix multiplication on CPU took " << fixed << setprecision(6) << cpu_elapsed_time << " seconds" << endl;

  // Allocate device memory
  float* d_A;
  float* d_B;
  float* d_C;
  nvtxRangePush("Cuda memory allocation");
  cudaMalloc(&d_A, sizeof(float) * M * K);
  cudaMalloc(&d_B, sizeof(float) * K * N);
  cudaMalloc(&d_C, sizeof(float) * M * N);
  nvtxRangePop();

  nvtxRangePush("Cuda memory host to device copy");
  // Copy data from host to device
  cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Set up grid and block dimensions
  dim3 block(32, 32);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // Launch kernel with timing
  clock_t gpu_start = clock();

  nvtxRangePush("Kernel");
  d_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  cudaDeviceSynchronize();
  nvtxRangePop();

  clock_t gpu_end = clock();
  double gpu_elapsed_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;

  cout << "Matrix multiplication on GPU took " << fixed << setprecision(6) << gpu_elapsed_time << " seconds" << endl;
  cout << "Speedup: " << fixed << setprecision(2) << cpu_elapsed_time / gpu_elapsed_time << "x" << endl;

  nvtxRangePush("Copy back to host");
  // Copy results back to host
  cudaMemcpy(h_result, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  verify_eq(h_result, h_C, M * N);
  cout << "Matrix multiplication completed successfully!" << endl;

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_result);

  nvtxRangePop();
  return 0;
}
