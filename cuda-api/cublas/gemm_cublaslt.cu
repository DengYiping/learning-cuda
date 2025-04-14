#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
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
  
  // Allocate host memory for large matrices using pinned memory
  float *A, *B, *h_cpu, *h_cublaslt_s, *h_cublaslt_h, *h_cublaslt_bf16;
  
  CHECK_CUDA(cudaMallocHost(&A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_cpu, 9 * sizeof(float)));  // Only need 3x3 for CPU verification
  CHECK_CUDA(cudaMallocHost(&h_cublaslt_s, M * N * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_cublaslt_h, M * N * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_cublaslt_bf16, M * N * sizeof(float)));

  if (!A || !B || !h_cpu || !h_cublaslt_s || !h_cublaslt_h || !h_cublaslt_bf16) {
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

  // cublasLt setup
  cublasLtHandle_t ltHandle;
  CHECK_CUBLAS(cublasLtCreate(&ltHandle));

  // Mem setup
  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // cublasLt SGEMM
  float alpha = 1.0f, beta = 0.0f;

  // Create matrix descriptors for B^T, A^T, C^T (treating row-major input as col-major transpose)
  // B (KxN row-major) -> B^T (NxK col-major), ld = N
  // A (MxK row-major) -> A^T (KxM col-major), ld = K
  // C (MxN row-major) -> C^T (NxM col-major), ld = N
  cublasLtMatrixLayout_t matB_T, matA_T, matC_T;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_T, CUDA_R_32F, N, K, N)); // B^T is NxK, leading dim N
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_T, CUDA_R_32F, K, M, K)); // A^T is KxM, leading dim K
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_T, CUDA_R_32F, N, M, N)); // C^T is NxM, leading dim N

  // Create operation descriptor
  cublasLtMatmulDesc_t operationDesc;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opN = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  // Create preference handle
  cublasLtMatmulPreference_t preference;
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspaceSize = 4 * 1024 * 1024;  // 4 MB, use size_t
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // Find the best algorithm
  int returnedAlgoCount = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, operationDesc, matB_T, matA_T, matC_T, matC_T, preference, 1, &heuristicResult, &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
    fprintf(stderr, "No algorithm returned\n");
    exit(EXIT_FAILURE);
  }

  // Execute cublasLt GEMM for FP32: Computes C^T = B^T * A^T
  CHECK_CUBLAS(cublasLtMatmul(
      ltHandle, operationDesc, &alpha,
      d_B, matB_T,  // B^T (data from d_B)
      d_A, matA_T,  // A^T (data from d_A)
      &beta,
      d_C, matC_T,  // C^T (output)
      d_C, matC_T,  // C^T (output)
      &heuristicResult.algo, NULL, 0, 0));

  CHECK_CUDA(cudaMemcpy(h_cublaslt_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost)); // M*N is correct size for C

  // For FP16 - create new matrices and descriptors
  half *h_A_h, *h_B_h;
  CHECK_CUDA(cudaMallocHost(&h_A_h, M * K * sizeof(half)));
  CHECK_CUDA(cudaMallocHost(&h_B_h, K * N * sizeof(half)));
  
  if (!h_A_h || !h_B_h) {
    fprintf(stderr, "Memory allocation for half precision input arrays failed\n");
    exit(EXIT_FAILURE);
  }
  
  half *d_A_h, *d_B_h;
  float *d_C_h;
  CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(float)));

  // Convert float to half
  for (int i = 0; i < M * K; i++) {
    h_A_h[i] = __float2half(A[i]);
  }
  for (int i = 0; i < K * N; i++) {
    h_B_h[i] = __float2half(B[i]);
  }
  CHECK_CUDA(cudaMemcpy(d_A_h, h_A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_h, h_B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

  // FP16 descriptors for B^T (FP16), A^T (FP16), C^T (FP32)
  cublasLtMatrixLayout_t matB_T_h, matA_T_h, matC_T_h;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_T_h, CUDA_R_16F, N, K, N)); // B^T is NxK (FP16), ld N
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_T_h, CUDA_R_16F, K, M, K)); // A^T is KxM (FP16), ld K
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_T_h, CUDA_R_32F, N, M, N)); // C^T is NxM (FP32), ld N

  // Create operation descriptor for FP16 inputs, FP32 compute/output/scale
  cublasLtMatmulDesc_t operationDesc_h;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc_h, CUBLAS_COMPUTE_32F, CUDA_R_32F)); // <-- Changed scale type to FP32
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc_h, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc_h, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  // Find the best algorithm for FP16: B^T * A^T -> C^T
  cublasLtMatmulHeuristicResult_t heuristicResult_h = {};
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, operationDesc_h, matB_T_h, matA_T_h, matC_T_h, matC_T_h, preference, 1, &heuristicResult_h, &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
    fprintf(stderr, "No algorithm returned for FP16\n");
    exit(EXIT_FAILURE);
  }

  // Execute cublasLt GEMM for FP16 inputs, FP32 output: Computes C^T = B^T * A^T
  CHECK_CUBLAS(cublasLtMatmul(
      ltHandle, operationDesc_h, &alpha,
      d_B_h, matB_T_h, // B^T (data from d_B_h, FP16)
      d_A_h, matA_T_h, // A^T (data from d_A_h, FP16)
      &beta,          // Use float beta
      d_C_h, matC_T_h, // C^T (output, FP32)
      d_C_h, matC_T_h, // C^T (output, FP32)
      &heuristicResult_h.algo, NULL, 0, 0));

  // Copy results back (already float)
  CHECK_CUDA(cudaMemcpy(h_cublaslt_h, d_C_h, M * N * sizeof(float), cudaMemcpyDeviceToHost)); // Copy directly to float host buffer

  // For BF16 - create new matrices and descriptors
  __nv_bfloat16 *h_A_bf16, *h_B_bf16;
  CHECK_CUDA(cudaMallocHost(&h_A_bf16, M * K * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMallocHost(&h_B_bf16, K * N * sizeof(__nv_bfloat16)));
  
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

  // BF16 descriptors for B^T, A^T, C^T
  // Inputs B^T (BF16), A^T (BF16), Output C^T (F32)
  cublasLtMatrixLayout_t matB_T_bf16, matA_T_bf16, matC_T_f32;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_T_bf16, CUDA_R_16BF, N, K, N)); // B^T (BF16) NxK, ld N
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_T_bf16, CUDA_R_16BF, K, M, K)); // A^T (BF16) KxM, ld K
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_T_f32, CUDA_R_32F, N, M, N)); // C^T (F32) NxM, ld N

  // Create operation descriptor for BF16 inputs, F32 compute/output
  cublasLtMatmulDesc_t operationDesc_bf16;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc_bf16, CUBLAS_COMPUTE_32F, CUDA_R_32F)); // Compute F32, Scale F32
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc_bf16, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc_bf16, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  // Find the best algorithm for BF16: B^T * A^T -> C^T
  cublasLtMatmulHeuristicResult_t heuristicResult_bf16 = {};
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, operationDesc_bf16, matB_T_bf16, matA_T_bf16, matC_T_f32, matC_T_f32, preference, 1, &heuristicResult_bf16, &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
    fprintf(stderr, "No algorithm returned for BF16\n");
    exit(EXIT_FAILURE);
  }

  // Execute cublasLt GEMM for BF16 inputs, F32 output: Computes C^T = B^T * A^T
  CHECK_CUBLAS(cublasLtMatmul(
      ltHandle, operationDesc_bf16, &alpha, // Use float alpha/beta
      d_B_bf16, matB_T_bf16, // B^T (data from d_B_bf16)
      d_A_bf16, matA_T_bf16, // A^T (data from d_A_bf16)
      &beta,                 // Use float alpha/beta
      d_C_bf16, matC_T_f32,  // C^T (output, F32)
      d_C_bf16, matC_T_f32,  // C^T (output, F32)
      &heuristicResult_bf16.algo, NULL, 0, 0));

  CHECK_CUDA(cudaMemcpy(h_cublaslt_bf16, d_C_bf16, M * N * sizeof(float), cudaMemcpyDeviceToHost)); // M*N is correct size for C

  // Print the results
  printf("CPU result (top-left 3x3 only):\n");
  PRINT_MATRIX(h_cpu, 3, 3);
  printf("cublasLt SGEMM result:\n");
  PRINT_MATRIX(h_cublaslt_s, M, N);
  printf("cublasLt HGEMM result:\n");
  PRINT_MATRIX(h_cublaslt_h, M, N);
  printf("cublasLt GEMM with BF16 result:\n");
  PRINT_MATRIX(h_cublaslt_bf16, M, N);

  // Free resources
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_T));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_T));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_T));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_T_h));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_T_h));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_T_h));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_T_bf16));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_T_bf16));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_T_f32));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc_h));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc_bf16));
  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtDestroy(ltHandle));

  // Free memory
  CHECK_CUDA(cudaFreeHost(A));
  CHECK_CUDA(cudaFreeHost(B));
  CHECK_CUDA(cudaFreeHost(h_cpu));
  CHECK_CUDA(cudaFreeHost(h_cublaslt_s));
  CHECK_CUDA(cudaFreeHost(h_cublaslt_h));
  CHECK_CUDA(cudaFreeHost(h_cublaslt_bf16));
  CHECK_CUDA(cudaFreeHost(h_A_h));
  CHECK_CUDA(cudaFreeHost(h_B_h));
  CHECK_CUDA(cudaFreeHost(h_A_bf16));
  CHECK_CUDA(cudaFreeHost(h_B_bf16));

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_A_h));
  CHECK_CUDA(cudaFree(d_B_h));
  CHECK_CUDA(cudaFree(d_C_h));
  CHECK_CUDA(cudaFree(d_A_bf16));
  CHECK_CUDA(cudaFree(d_B_bf16));
  CHECK_CUDA(cudaFree(d_C_bf16));

  return 0;
} 