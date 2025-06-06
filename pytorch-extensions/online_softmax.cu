#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/limits>
// #include <cub/cub.cuh>
#include <cmath>
#include <algorithm>
#include <torch/extension.h>

#include "online_softmax.hpp"

unsigned int next_power_of_2_min_32(unsigned int n) {
  // Handle edge case for 0 or negative (though unsigned int prevents negative)
  // If n is 0, the next power of 2 is 1.
  // If n is already a power of 2, return n.
  if (n > 0 && (n & (n - 1)) == 0) {
    return std::max(n, 32u); // Use 32u to ensure it's an unsigned int literal
  }

  // Find the next power of 2 using bit manipulation
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  // For 64-bit integers, you'd add:
  // n |= n >> 32;
  n++;

  // Ensure the result is at least 32
  return std::max(n, 32u);
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void online_softmax_kernel(
  const scalar_t* __restrict__ x,
  scalar_t* __restrict__ output,
  int input_row_stride, int input_col_stride,
  int output_row_stride, int output_col_stride,
  int M, int N
) {
  // This is the kernel to implement
  __shared__ scalar_t sram_m[BLOCK_SIZE];
  __shared__ scalar_t sram_d[BLOCK_SIZE];

  int idx = blockIdx.x * input_row_stride + threadIdx.x * input_col_stride;

  // Init sram
  scalar_t val = cuda::std::numeric_limits<scalar_t>::lowest();
  scalar_t agg = scalar_t(0);

  if (threadIdx.x < N) {
    val = x[idx];
    agg = scalar_t(1);
  }

  sram_m[threadIdx.x] = val;
  sram_d[threadIdx.x] = agg;
  __syncthreads();

  // parallel reduction once
  for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
    if (threadIdx.x < stride) {
      scalar_t left = sram_m[threadIdx.x];
      scalar_t right = sram_m[threadIdx.x + stride];
      scalar_t m = fmaxf(left, right);
      scalar_t d = sram_d[threadIdx.x] * expf(left - m) + sram_d[threadIdx.x + stride] * expf(right - m);
      sram_m[threadIdx.x] = m;
      sram_d[threadIdx.x] = d;
    }
    __syncthreads();
  }

  // calculate value
  if (threadIdx.x < N) {
    scalar_t final_val = expf(val - sram_m[0]) / sram_d[0];
    int output_idx = blockIdx.x * output_row_stride + threadIdx.x * output_col_stride;
    output[output_idx] = final_val;
  }
}

torch::Tensor online_softmax(torch::Tensor x) {
  TORCH_CHECK(x.dim() == 2, "safe_softmax_cuda: requires a 2D tensor");
  auto output = torch::empty_like(x);

  int64_t M = x.size(0);
  int64_t N = x.size(1);

  // PyTorch strides are in element‐counts
  int input_row_stride  = x.stride(0);
  int input_col_stride  = x.stride(1);
  int output_row_stride = output.stride(0);
  int output_col_stride = output.stride(1);

  // one block per row, up to 2048 threads (one per column)
  int threads = next_power_of_2_min_32(N);
  int blocks  = M;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "online_softmax_cuda", [&] {
    // branch on threads to fill the BLOCK_SIZE template parameter
    switch (threads) {
      case 32:
        online_softmax_kernel<scalar_t, 32>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      case 64:
        online_softmax_kernel<scalar_t, 64>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      case 128:
        online_softmax_kernel<scalar_t, 128>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      case 256:
        online_softmax_kernel<scalar_t, 256>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      case 512:
        online_softmax_kernel<scalar_t, 512>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      case 1024:
        online_softmax_kernel<scalar_t, 1024>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      case 2048:
        online_softmax_kernel<scalar_t, 2048>
          <<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_row_stride, input_col_stride,
            output_row_stride, output_col_stride,
            M, N);
        break;
      default:
        AT_ERROR("online_softmax_cuda: unsupported block size ", threads);
    }
  });

  // optional: check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("safe_softmax kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  return output;
}

