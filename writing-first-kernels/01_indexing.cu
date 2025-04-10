#include <stdio.h>

__forceinline__ __device__ unsigned d_get_lane_id()
{
  unsigned ret;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__global__ void whoami(void) {
  // blockIdx store the current index of the block
  // gridDim stores the information about grid setup (a.k.a how many blocks)
  int d_block_id = blockIdx.x +
    blockIdx.y * gridDim.x +
    blockIdx.z * gridDim.x * gridDim.y;

  int d_block_offset = d_block_id * blockDim.x * blockDim.y * blockDim.z;

  // threadIdx store the current index of the thread
  // blockDim stores the information about the block setup (a.k.a how many threads in a block)
  int d_thread_offset = threadIdx.x +
    threadIdx.y * blockDim.x +
    threadIdx.z * blockDim.x * blockDim.y;

  int d_lane_id = d_thread_offset & 0x1f;

  int d_id = d_thread_offset + d_block_offset;

  printf("%04d | Block(%d %d %d) = %3d | Lane(%2d)=%2d | Thread(%d %d %d) = %3d\n",
         d_id,
         blockIdx.x,
         blockIdx.y,
         blockIdx.z,
         d_block_id,
         d_lane_id,
         d_get_lane_id(),
         threadIdx.x,
         threadIdx.y,
         threadIdx.z,
         d_thread_offset
         );
}

int main(int argc, char** argv) {
  // Use a large grid that is a multiple of 132
  const int b_x = 2 * 2, b_y = 3, b_z = 11;
  const int t_x = 4, t_y = 4, t_z = 4;

  dim3 blocks(b_x, b_y, b_z);
  dim3 threadsPerBlock(t_x, t_y, t_z);

  whoami<<<blocks, threadsPerBlock>>>();
}
