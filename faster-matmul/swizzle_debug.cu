#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include "ptx.cuh" // Assuming this file exists and contains necessary PTX wrappers
#include <stdio.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib> // For exit()
#include <numeric> // For std::iota
#include <algorithm> // For std::min
#include <cmath> // For std::fabs

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Runtime error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUDA_DRIVER(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *err_str; \
        cuGetErrorString(err, &err_str); \
        fprintf(stderr, "CUDA Driver error in %s at line %d: %s\n", __FILE__, __LINE__, err_str); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

namespace cg = cooperative_groups;

// N is the width (K) of the tile in shared memory
template <int N>
__device__ __forceinline__ float tma_swizzle_128b_load(const float* smem_ptr, uint y, uint x) {
    // 128b swizzle with 16 byte chunks (4 floats)
    int swizzled_index = y * N + ((y % 8) ^ (x / 4)) * 4 + (x % 4);

    return smem_ptr[swizzled_index];
}

// Kernel to load data using TMA with swizzling, check the swizzle decode, and store results
// M = rows, K = cols for the loaded tile
template <int M, int K, int K_GLOBAL>
__global__ void tma_load_debug_kernel(
    const __grid_constant__ CUtensorMap tensor_map_swizzled,
    float* dest_swizzled,
    int* d_error_count)
{
    // --- Static MBarrier ---
    // Statically allocate the mbarrier, ensuring 16-byte alignment.
    alignas(16) __shared__ uint64_t mbarrier_s[1];
    uint64_t* mbar = mbarrier_s;

    // --- Dynamic Shared Memory for Data Buffers ---
    alignas(1024) extern __shared__ float smem_dyn[];

    // --- TMA Setup ---
    const int tid = threadIdx.x;
    const int num_threads_block = blockDim.x;
    const bool is_master_thread = (tid == 0);
    constexpr size_t bytes_to_copy = M * K * sizeof(float);

    // Initialize the mbarrier from thread 0.
    if (is_master_thread) {
        ptx::mbarrier_init(mbar, num_threads_block); // Use blockDim.x
        ptx::fence_mbarrier_init_release_cluster();
    }
    __syncthreads(); // Ensure barrier is initialized before use

    // --- Perform TMA Load ---
    if (is_master_thread) {
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_dyn),
            (const uint64_t*)&tensor_map_swizzled,
            0, 0,
            mbar
        );
        ptx::mbarrier_arrive_expect_tx(mbar, bytes_to_copy);
    } else {
        ptx::mbarrier_arrive(mbar);
    }

    // Wait for TMA copy to complete.
    ptx::mbarrier_wait_parity(mbar, 0);
    __syncthreads(); // Ensure visibility of smem_dyn writes across threads

    // --- Check Swizzle Decode ---
    int total_elements = M * K;
    int thread_idx = threadIdx.x;
    int num_threads = blockDim.x;
    const float tolerance = 1e-6f; // Tolerance for float comparison

    for (int i = thread_idx; i < total_elements; i += num_threads) {
        int row = i / K; // Logical row in the M x K tile
        int col = i % K; // Logical col in the M x K tile

        // Read value from swizzled shared memory using the decode function
        float read_value = tma_swizzle_128b_load<K>(smem_dyn, row, col);

        // Calculate the expected value (linear index in the original global matrix)
        float expected_value = (float)(row * K_GLOBAL + col); // Use K_GLOBAL

        // Compare - use tolerance for floating point
        if (fabsf(read_value - expected_value) > tolerance) {
            // Atomically increment the error counter in global memory
            atomicAdd(d_error_count, 1);
        }
    }
    __syncthreads(); // Ensure all threads finish checking before copying


    // --- Copy from Shared Memory to Global Memory ---
    // All threads in the block cooperate to copy the data from shared memory buffers
    // to the corresponding global memory destination buffers.
    // Use a grid-stride loop pattern for copying elements.
    for (int i = thread_idx; i < total_elements; i += num_threads) {
        dest_swizzled[i] = smem_dyn[i];
    }
    __syncthreads();
}

// Helper function to print the entire matrix stored in row-major order.
void print_matrix(const std::string& title, const float* matrix, int rows, int cols) {
    std::cout << "\n--- " << title << " (" << rows << "x" << cols << ", Strided by 4 cols) ---" << std::endl;
    // Use the full dimensions of the matrix.
    int max_r = rows;
    int max_c = cols;

    // Print column headers for alignment, striding by 4
    std::cout << "      "; // Space for row index
    for (int j = 0; j < max_c; j += 4) {
         int end_col = std::min(j + 3, max_c - 1);
         std::string header = "C" + std::to_string(j) + "-" + std::to_string(end_col);
         // Adjust width based on header length, adding padding
         std::cout << std::left << std::setw(header.length() + 2) << header;
    }
    std::cout << std::right << std::endl; // Reset alignment
     std::cout << "      "; // Space for row index separator
     for (int j = 0; j < max_c; j += 4) {
         int end_col = std::min(j + 3, max_c - 1);
         std::string header = "C" + std::to_string(j) + "-" + std::to_string(end_col);
         std::string separator(header.length(), '-');
         // Adjust width based on header length, adding padding
         std::cout << std::left << std::setw(header.length() + 2) << separator;
    }
    std::cout << std::right << std::endl; // Reset alignment

    // Iterate through the rows and columns to print, striding by 4 columns.
    for (int i = 0; i < max_r; ++i) {
        std::cout << "R" << std::setw(2) << i << " | "; // Print row index
        for (int j = 0; j < max_c; j += 4) {
            int end_col = std::min(j + 3, max_c - 1);
            std::string header = "C" + std::to_string(j) + "-" + std::to_string(end_col);
            // Access the element using row-major indexing: index = row * num_cols + col
            // Print the value at the start of the 4-column block
            std::cout << std::left << std::fixed << std::setw(header.length() + 2) << std::setprecision(0) << matrix[i * cols + j];
        }
        std::cout << std::right << std::endl; // Newline after each row and reset alignment
    }
    std::cout << "--------------------------------------------------" << std::endl;
}


int main() {
    // --- Configuration ---
    const int M = 64; // Tile/Box rows
    const int K = 32; // Tile/Box columns
    const int M_GLOBAL = 64; // Global Matrix rows (larger than M)
    const int K_GLOBAL = 32; // Global Matrix cols (larger than K)

    // Kernel launch parameters
    const int BLOCK_DIM_X = 256; // Number of threads per block (must be <= 1024)

    // --- CUDA Initialization and Capability Check ---
    CHECK_CUDA_DRIVER(cuInit(0)); // Initialize CUDA Driver API
    CUcontext ctx;
    CUdevice dev;
    CHECK_CUDA_DRIVER(cuDeviceGet(&dev, 0));       // Get device 0
    CHECK_CUDA_DRIVER(cuCtxCreate(&ctx, 0, dev)); // Create context for the device

    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Device: " << deviceProp.name << std::endl;

    // Check for required TMA/mbarrier features
    int cooperativeLaunch, clusterLaunch, asyncEngineCount;
    CHECK_CUDA(cudaDeviceGetAttribute(&cooperativeLaunch, cudaDevAttrCooperativeLaunch, 0));
    CHECK_CUDA(cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, 0));
    CHECK_CUDA(cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, 0));

    std::cout << "Supports Cooperative Launch: " << (cooperativeLaunch ? "Yes" : "No") << std::endl;
    std::cout << "Supports Cluster Launch:     " << (clusterLaunch ? "Yes" : "No") << std::endl;
    std::cout << "Async Copy Engines:        " << asyncEngineCount << std::endl;

    // NOTE: Standard launch is used, but mbarrier requires cluster/async features.
    // This configuration might not work correctly on all systems/drivers.
    if (!clusterLaunch || asyncEngineCount == 0) {
        std::cerr << "Error: Device lacks required features for TMA mbarrier operations." << std::endl;
        CHECK_CUDA_DRIVER(cuCtxDestroy(ctx));
        return EXIT_FAILURE;
    }

    // --- Host Data Allocation and Initialization ---
    size_t global_matrix_elements = M_GLOBAL * K_GLOBAL;
    size_t global_matrix_size_bytes = global_matrix_elements * sizeof(float);
    std::vector<float> h_A(global_matrix_elements);

    // Fill host matrix with ordered numbers 0.0, 1.0, 2.0, ...
    std::iota(h_A.begin(), h_A.end(), 0.0f);

    // --- Device Data Allocation ---
    size_t tile_matrix_elements = M * K;
    size_t tile_matrix_size_bytes = tile_matrix_elements * sizeof(float);
    float* d_A = nullptr;
    float* d_smem_swizzled = nullptr;
    int* d_error_count = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, global_matrix_size_bytes)); // Allocate global size
    CHECK_CUDA(cudaMalloc(&d_smem_swizzled, tile_matrix_size_bytes));   // Allocate tile size for result
    CHECK_CUDA(cudaMalloc(&d_error_count, sizeof(int))); // Allocate error counter
    CHECK_CUDA(cudaMemset(d_error_count, 0, sizeof(int))); // Initialize error counter to 0

    // --- Copy Host Data to Device ---
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), global_matrix_size_bytes, cudaMemcpyHostToDevice));

    // --- Tensor Map Creation ---
    CUtensorMap tensor_map_swizzled;

    // TMA expects dimensions ordered {innermost, ... outermost}
    const uint64_t globalDimA[] = {(uint64_t)K_GLOBAL, (uint64_t)M_GLOBAL}; // Use GLOBAL dimensions
    // Strides: {stride within innermost dim, stride between elements in next dim, ...}
    const uint64_t globalStrideA[] = {(uint64_t)K_GLOBAL * sizeof(float)}; // Use GLOBAL K stride
    // Box dimensions define the shape of the tile being mapped.
    const cuuint32_t boxDimA[] = {K, M}; // Use TILE dimensions {32, 64}
    // Element strides within the float (only relevant for sub-byte types)
    const cuuint32_t elementStrides[] = {1, 1}; // Stride within element (size 1 for float)

    // Create tensor map WITH 128B swizzling
    CHECK_CUDA_DRIVER(cuTensorMapEncodeTiled(
        &tensor_map_swizzled,             // Output tensor map object
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,  // Data type
        2,                                // Rank
        (void*)d_A,                       // Global memory address
        globalDimA,                       // Global dimensions
        globalStrideA,                    // Global strides (outer stride ptr)
        boxDimA,                          // Box dimensions
        elementStrides,                   // Element stride
        CU_TENSOR_MAP_INTERLEAVE_NONE,    // No interleaving
        CU_TENSOR_MAP_SWIZZLE_128B,       // 128B SWIZZLE applied
        CU_TENSOR_MAP_L2_PROMOTION_NONE,  // L2 promotion
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // OOB fill mode
    ));

    // --- Kernel Launch ---
    dim3 blockDim(BLOCK_DIM_X); // Threads per block
    dim3 gridDim(1);            // Single block for cooperative launch

    // Calculate required dynamic shared memory for the single swizzled buffer
    size_t smem_bytes_dynamic = M * K * sizeof(float);
    // Static shared memory (mbarrier) is allocated statically within the kernel
    size_t smem_bytes_static = sizeof(uint64_t);

    std::cout << "Required Dynamic Shared Memory: " << smem_bytes_dynamic << " bytes" << std::endl;
    std::cout << "Static Shared Memory (mbarrier): " << smem_bytes_static << " bytes" << std::endl;

    // Check if requested shared memory exceeds device limits
    if (smem_bytes_dynamic + smem_bytes_static > deviceProp.sharedMemPerBlock) {
         std::cerr << "Error: Required shared memory (" << smem_bytes_dynamic + smem_bytes_static
                   << " bytes) exceeds device limit per block (" << deviceProp.sharedMemPerBlock
                   << " bytes)." << std::endl;
          CHECK_CUDA(cudaFree(d_A));
          CHECK_CUDA(cudaFree(d_smem_swizzled));
          CHECK_CUDA(cudaFree(d_error_count));
          CHECK_CUDA_DRIVER(cuCtxDestroy(ctx));
        return EXIT_FAILURE;
    }

    std::cout << "Launching standard kernel..." << std::endl;
    tma_load_debug_kernel<M, K, K_GLOBAL><<<gridDim, blockDim, smem_bytes_dynamic, 0>>>(
        tensor_map_swizzled,
        d_smem_swizzled,
        d_error_count
    );

    // Ensure kernel completion and check for launch errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    // --- Copy Results Back to Host ---
    std::vector<float> h_smem_swizzled(tile_matrix_elements);   // Tile size
    int h_error_count = 0; // Host variable for error count

    CHECK_CUDA(cudaMemcpy(h_smem_swizzled.data(), d_smem_swizzled, tile_matrix_size_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_error_count, d_error_count, sizeof(int), cudaMemcpyDeviceToHost)); // Copy error count back

    // --- Print Results ---
    std::cout << "--- Swizzle Decode Check ---" << std::endl;
    std::cout << "Number of mismatches found: " << h_error_count << std::endl;
    if (h_error_count == 0) {
        std::cout << "Swizzle decode function appears to be correct." << std::endl;
    } else {
        std::cout << "Swizzle decode function INCORRECT." << std::endl;
    }
    // Print the original h_A (using M_GLOBAL, K_GLOBAL dimensions)
    print_matrix("Original Host Data (h_A)", h_A.data(), M_GLOBAL, K_GLOBAL);
    // Print the results (using M, K tile dimensions)
    print_matrix("TMA Load 128B Swizzle (Simulated Shared Memory)", h_smem_swizzled.data(), M, K);

    // --- Cleanup ---
    std::cout << "Cleaning up device memory..." << std::endl;
    CHECK_CUDA(cudaFree(d_A)); // Free global allocation
    CHECK_CUDA(cudaFree(d_smem_swizzled));   // Free tile-sized allocation
    CHECK_CUDA(cudaFree(d_error_count));     // Free error counter
    CHECK_CUDA_DRIVER(cuCtxDestroy(ctx)); // Destroy the CUDA context

    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
} 