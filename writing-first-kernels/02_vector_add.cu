#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cstring>

#define N 100000000 // Vector size 100 million
#define BLOCK_SIZE 256 // Number of blocks

void h_vector_add(float* vec1, float* vec2, float* result, int len) {
  for (int i = 0; i < len; i++) {
    result[i] = vec1[i] + vec2[i];
  }
}

__global__ void d_vector_add(float* d_vec1, float* d_vec2, float* d_result, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    d_result[i] = d_vec1[i] + d_vec2[i];
  }
}

// New kernel using float4 for better memory throughput
__global__ void d_vector_add_float4(float* d_vec1, float* d_vec2, float* d_result, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i4 = i * 4; // Each thread processes 4 elements

  if (i4 + 3 < len) { // Make sure we have 4 elements to process
    float4 v1 = reinterpret_cast<float4*>(d_vec1)[i];
    float4 v2 = reinterpret_cast<float4*>(d_vec2)[i];

    float4 res;
    res.x = v1.x + v2.x;
    res.y = v1.y + v2.y;
    res.z = v1.z + v2.z;
    res.w = v1.w + v2.w;

    reinterpret_cast<float4*>(d_result)[i] = res;
  }
  else if (i4 < len) { // Handle remaining elements
    for (int j = 0; j < len - i4; j++) {
      d_result[i4 + j] = d_vec1[i4 + j] + d_vec2[i4 + j];
    }
  }
}

// New kernel using float4 with proper padding
__global__ void d_vector_add_float4_padded(float* d_vec1, float* d_vec2, float* d_result, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // No boundary check needed since input is padded
  float4 v1 = reinterpret_cast<float4*>(d_vec1)[i];
  float4 v2 = reinterpret_cast<float4*>(d_vec2)[i];

  float4 res;
  res.x = v1.x + v2.x;
  res.y = v1.y + v2.y;
  res.z = v1.z + v2.z;
  res.w = v1.w + v2.w;

  reinterpret_cast<float4*>(d_result)[i] = res;
}

void h_init_vector(float* vec, int size) {
  for (int i = 0; i < size; i++) {
    vec[i] = (float) rand() / RAND_MAX;
  }
}

double h_norm(float* vec, int size) {
  double result = 0.0;

  for (int i =0; i < size; i++) {
    result += vec[i] * vec[i];
  }

  return sqrt(result);
}


int main(int argc, char** argv) {
  float* h_vec1 = (float*) malloc(sizeof(float) * N);
  float* h_vec2 = (float*) malloc(sizeof(float) * N);
  float* h_result = (float*) malloc(sizeof(float) * N);
  h_init_vector(h_vec1, N);
  h_init_vector(h_vec2, N);

  clock_t start = clock();
  h_vector_add(h_vec1, h_vec2, h_result, N);
  clock_t end = clock();
  double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("Vector addition on CPU took %.6f seconds\n", elapsed_time);
  printf("Norm: %.03f\n", h_norm(h_result, N));

  std::memset(h_result, 0, sizeof(float) * N);
  // ======== GPU computation
  float* d_vec1;
  float* d_vec2;
  float* d_result;

  cudaMalloc(&d_vec1, sizeof(float) * N);
  cudaMalloc(&d_vec2, sizeof(float) * N);
  cudaMalloc(&d_result, sizeof(float) * N);

  cudaMemcpy(d_vec1, h_vec1, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec2, h_vec2, sizeof(float) * N, cudaMemcpyHostToDevice);


  // Ceil (N / BLOCK_SIZE)
  int d_num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  start = clock();
  d_vector_add<<<d_num_blocks, BLOCK_SIZE>>>(d_vec1, d_vec2, d_result, N);
  cudaDeviceSynchronize(); // Wait for kernel to finish
  end = clock();
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

  cudaMemcpy(h_result, d_result, sizeof(float) * N, cudaMemcpyDeviceToHost);
  printf("Vector addition on GPU took %.6f seconds\n", elapsed_time);
  printf("Norm: %.03f\n", h_norm(h_result, N));

  // Benchmark float4 version
  std::memset(h_result, 0, sizeof(float) * N);
  cudaMemset(d_result, 0, sizeof(float) * N);

  // For float4, each thread processes 4 elements, so we need fewer blocks
  int d_num_blocks_float4 = (N/4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  start = clock();
  d_vector_add_float4<<<d_num_blocks_float4, BLOCK_SIZE>>>(d_vec1, d_vec2, d_result, N);
  cudaDeviceSynchronize(); // Wait for kernel to finish
  end = clock();
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

  cudaMemcpy(h_result, d_result, sizeof(float) * N, cudaMemcpyDeviceToHost);
  printf("Vector addition on GPU (float4) took %.6f seconds\n", elapsed_time);
  printf("Norm: %.03f\n", h_norm(h_result, N));

  // Benchmark padded float4 version
  std::memset(h_result, 0, sizeof(float) * N);
  cudaMemset(d_result, 0, sizeof(float) * N);

  // Calculate padded size (multiple of 4)
  int padded_size = ((N + 3) / 4) * 4;
  float* d_vec1_padded;
  float* d_vec2_padded;
  float* d_result_padded;

  cudaMalloc(&d_vec1_padded, sizeof(float) * padded_size);
  cudaMalloc(&d_vec2_padded, sizeof(float) * padded_size);
  cudaMalloc(&d_result_padded, sizeof(float) * padded_size);

  // Copy data and zero-pad
  cudaMemcpy(d_vec1_padded, h_vec1, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec2_padded, h_vec2, sizeof(float) * N, cudaMemcpyHostToDevice);
  if (padded_size > N) {
    cudaMemset(d_vec1_padded + N, 0, sizeof(float) * (padded_size - N));
    cudaMemset(d_vec2_padded + N, 0, sizeof(float) * (padded_size - N));
  }

  int d_num_blocks_padded = padded_size / 4 / BLOCK_SIZE;
  if (padded_size / 4 % BLOCK_SIZE != 0) d_num_blocks_padded++;

  start = clock();
  d_vector_add_float4_padded<<<d_num_blocks_padded, BLOCK_SIZE>>>(d_vec1_padded, d_vec2_padded, d_result_padded, padded_size);
  cudaDeviceSynchronize(); // Wait for kernel to finish
  end = clock();
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

  cudaMemcpy(h_result, d_result_padded, sizeof(float) * N, cudaMemcpyDeviceToHost);
  printf("Vector addition on GPU (padded float4) took %.6f seconds\n", elapsed_time);
  printf("Norm: %.03f\n", h_norm(h_result, N));

  cudaFree(d_vec1_padded);
  cudaFree(d_vec2_padded);
  cudaFree(d_result_padded);
  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_result);

  free(h_vec1);
  free(h_vec2);
  free(h_result);
}
