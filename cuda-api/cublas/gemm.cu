#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

#undef PRINT_MATRIX
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
  
  // Allocate host memory for large matrices
  float *A = (float*)malloc(M * K * sizeof(float));
  float *B = (float*)malloc(K * N * sizeof(float));
  float *h_cpu = (float*)malloc(9 * sizeof(float));  // Only need 3x3 for CPU verification
  float *h_cublas_s = (float*)malloc(M * N * sizeof(float));
  float *h_cublas_h = (float*)malloc(M * N * sizeof(float));
  float *h_cublas_bf16 = (float*)malloc(M * N * sizeof(float));

  if (!A || !B || !h_cpu || !h_cublas_s || !h_cublas_h || !h_cublas_bf16) {
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

  // cuBLAS setup
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Mem setup
  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // cuBLAS SGEMM
  float alpha = 1.0f, beta = 0.0f;

  // cuBLAS are in column major. We can get around this using
  // A @ B = C => B.T @ A.T = C.T
  // row major matrix's transpose is the same matrix in the column major format

  // cublasStatus_t cublasSgemm(cublasHandle_t handle,
  //                            cublasOperation_t transa, cublasOperation_t transb,
  //                            int m, int n, int k,
  //                            const float           *alpha,
  //                            const float           *A, int lda,
  //                            const float           *B, int ldb,
  //                            const float           *beta,
  //                            float           *C, int ldc)
  CHECK_CUBLAS(cublasSgemm(handle,
                           CUBLAS_OP_N,CUBLAS_OP_N,
                           N, // matrix's B's column
                           M, // matrix's A's rows
                           K, // not changed in the row major format
                           &alpha,
                           d_B, N, // matrix's B's column
                           d_A, K, // matrix's A's column
                           &beta,
                           d_C, N // matrix's C's column
                           )
               );
  CHECK_CUDA(cudaMemcpy(h_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // cuBLAS HGEMM - Allocate on heap instead of stack
  half *h_A_h = (half*)malloc(M * K * sizeof(half));
  half *h_B_h = (half*)malloc(K * N * sizeof(half));
  half *h_C_h = (half*)malloc(M * N * sizeof(half));
  if (!h_A_h || !h_B_h || !h_C_h) {
    fprintf(stderr, "Memory allocation for half precision arrays failed\n");
    exit(EXIT_FAILURE);
  }
  
  half *d_A_h, *d_B_h, *d_C_h;
  CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

  // convert float to half
  for (int i = 0; i < M * K; i++) {
    h_A_h[i] = __float2half(A[i]);
  }
  for (int i = 0; i < K * N; i++) {
    h_B_h[i] = __float2half(B[i]);
  }
  CHECK_CUDA(cudaMemcpy(d_A_h, h_A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_h, h_B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

  half alpha_h = __float2half(1.0f);
  half beta_h = __float2half(0.0f);

  CHECK_CUBLAS(cublasHgemm(handle,
                           CUBLAS_OP_N,CUBLAS_OP_N,
                           N, // matrix's B's column
                           M, // matrix's A's rows
                           K, // not changed in the row major format
                           &alpha_h,
                           d_B_h, N, // matrix's B's column
                           d_A_h, K, // matrix's A's column
                           &beta_h,
                           d_C_h, N // matrix's C's column
                           )
               );
  CHECK_CUDA(cudaMemcpy(h_C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));

  // convert half to float
  for (int i = 0; i < M * N; i++) {
    h_cublas_h[i] = __half2float(h_C_h[i]);
  }

  // cuBLAS GemmEx with BF16 - Allocate on heap instead of stack
  __nv_bfloat16 *h_A_bf16 = (__nv_bfloat16*)malloc(M * K * sizeof(__nv_bfloat16));
  __nv_bfloat16 *h_B_bf16 = (__nv_bfloat16*)malloc(K * N * sizeof(__nv_bfloat16));
  if (!h_A_bf16 || !h_B_bf16) {
    fprintf(stderr, "Memory allocation for bfloat16 arrays failed\n");
    exit(EXIT_FAILURE);
  }
  
  __nv_bfloat16 *d_A_bf16, *d_B_bf16;
  float *d_C_bf16; // Using float for output

  CHECK_CUDA(cudaMalloc(&d_A_bf16, M * K * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_B_bf16, K * N * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_C_bf16, M * N * sizeof(float)));

  // Convert float to bfloat16
  for (int i = 0; i < M * K; i++) {
    h_A_bf16[i] = __float2bfloat16(A[i]);
  }
  for (int i = 0; i < K * N; i++) {
    h_B_bf16[i] = __float2bfloat16(B[i]);
  }

  CHECK_CUDA(cudaMemcpy(d_A_bf16, h_A_bf16, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_bf16, h_B_bf16, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  // cublasGemmEx with bfloat16
  alpha = 1.0f;
  beta = 0.0f;

  // cublasStatus_t cublasGemmEx(cublasHandle_t handle,
  //                           cublasOperation_t transa, cublasOperation_t transb,
  //                           int m, int n, int k,
  //                           const void *alpha,
  //                           const void *A, cudaDataType_t Atype, int lda,
  //                           const void *B, cudaDataType_t Btype, int ldb,
  //                           const void *beta,
  //                           void *C, cudaDataType_t Ctype, int ldc,
  //                           cudaDataType_t computeType,
  //                           cublasGemmAlgo_t algo)
  CHECK_CUBLAS(cublasGemmEx(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           N, // matrix B's column
                           M, // matrix A's rows
                           K, // not changed in the row major format
                           &alpha,
                           d_B_bf16, CUDA_R_16BF, N, // matrix B's column
                           d_A_bf16, CUDA_R_16BF, K, // matrix A's column
                           &beta,
                           d_C_bf16, CUDA_R_32F, N, // matrix C's column
                           CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT));

  CHECK_CUDA(cudaMemcpy(h_cublas_bf16, d_C_bf16, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // print the result
  printf("CPU result (top-left 3x3 only):\n");
  PRINT_MATRIX(h_cpu, 3, 3);
  printf("cuBLAS SGEMM result:\n");
  PRINT_MATRIX(h_cublas_s, M, N);
  printf("cuBLAS HGEMM result:\n");
  PRINT_MATRIX(h_cublas_h, M, N);
  printf("cuBLAS GemmEx with BF16 result:\n");
  PRINT_MATRIX(h_cublas_bf16, M, N);

  // free the memory
  free(A);
  free(B);
  free(h_cpu);
  free(h_cublas_s);
  free(h_cublas_h);
  free(h_cublas_bf16);
  free(h_A_h);
  free(h_B_h);
  free(h_C_h);
  free(h_A_bf16);
  free(h_B_bf16);
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_A_h));
  CHECK_CUDA(cudaFree(d_B_h));
  CHECK_CUDA(cudaFree(d_C_h));
  CHECK_CUDA(cudaFree(d_A_bf16));
  CHECK_CUDA(cudaFree(d_B_bf16));
  CHECK_CUDA(cudaFree(d_C_bf16));
  CHECK_CUBLAS(cublasDestroy(handle));

  return 0;
}
