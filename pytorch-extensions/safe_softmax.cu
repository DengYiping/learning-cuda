#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "safe_softmax.hpp"

template <typename scalar_t>
__global__ void safe_softmax_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    int input_row_stride, int input_col_stride,
    int output_row_stride, int output_col_stride,
    int M, int N
) {
  // each block handles one row
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // global column index this thread works on
  int col = tid;

  // declare shared memory for reduction (max and then sum)
  __shared__ scalar_t sdata[1024];

  // 1) compute row-max
  scalar_t val = -std::numeric_limits<scalar_t>::infinity();
  if (col < N) {
    val = x[row * input_row_stride + col * input_col_stride];
  }
  sdata[tid] = val;
  __syncthreads();

  // parallel reduction for max
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if constexpr (std::is_same_v<scalar_t, float>) {
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
      } else if constexpr (std::is_same_v<scalar_t, double>) {
        sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
      } else {
        sdata[tid] = std::max(sdata[tid], sdata[tid + stride]); // Fallback for other types if any
      }
    }
    __syncthreads();
  }

  // 2) compute exponentials and row‐sum
  scalar_t ex;
  if constexpr (std::is_same_v<scalar_t, float>) {
    ex = __expf(val - sdata[0]);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    ex = exp(val - sdata[0]);
  } else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    ex = hexp(val - sdata[0]);
  } else if constexpr (std::is_same_v<scalar_t, __half>) {
    ex = hexp(val - sdata[0]);
  } else {
    ex = exp(val - sdata[0]);
  }

  if (col < N) {
    sdata[tid] = ex;
  } else {
    sdata[tid] = scalar_t(0);
  }
  __syncthreads();

  // parallel reduction for sum
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  scalar_t row_sum = sdata[0];

  // 3) write out normalized result
  if (col < N) {
    output[row * output_row_stride + col * output_col_stride] = ex / row_sum;
  }
}

torch::Tensor safe_softmax(torch::Tensor x) {
  TORCH_CHECK(x.dim() == 2, "safe_softmax: requires a 2D tensor");
  auto output = torch::empty_like(x);

  int64_t M = x.size(0);
  int64_t N = x.size(1);

  // PyTorch strides are in element‐counts
  int input_row_stride  = x.stride(0);
  int input_col_stride  = x.stride(1);
  int output_row_stride = output.stride(0);
  int output_col_stride = output.stride(1);

  // one block per row, up to 1024 threads (one per column)
  int threads = std::min<int64_t>(1024, N);
  int blocks  = M;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "safe_softmax", [&] {
    safe_softmax_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        input_row_stride, input_col_stride,
        output_row_stride, output_col_stride,
        M, N
    );
  });

  // optional: check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("safe_softmax kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  return output;
}

