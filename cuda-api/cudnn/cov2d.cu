#include <iostream>       // For std::cerr
#include <string>         // For std::string
#include <cstdlib>        // For exit, EXIT_FAILURE
#include <cuda_runtime.h> // For CUDA types and functions
#include <cudnn.h>        // For cuDNN types and functions
#include <iomanip>        // For std::setw, std::setprecision
#include <random>         // For random number generation

// Define dimensions for CNN
constexpr int BATCH_SIZE = 256;
constexpr int CHANNELS = 32;
constexpr int HEIGHT = 224;
constexpr int WIDTH = 224;
constexpr int KERNEL_HEIGHT = 3;
constexpr int KERNEL_WIDTH = 3;
constexpr int STRIDE = 1;
constexpr int PADDING = 1;
constexpr int OUTPUT_CHANNELS = 2;

// Helper function for CUDA error checking
inline void checkCudaResult(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << file << "' in line " << line << " : "
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
        // Terminate program on error, similar to the original macro
        exit(EXIT_FAILURE);
    }
}

// Helper function for cuDNN error checking
inline void checkCudnnResult(cudnnStatus_t err, const char* file, int line) {
    if (err != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN error in file '" << file << "' in line " << line << " : "
                  << cudnnGetErrorString(err) << " (" << err << ")" << std::endl;
        // Terminate program on error, similar to the original macro
        exit(EXIT_FAILURE);
    }
}

// Define macros that call the inline helper functions
// These macros preserve the convenient call syntax and automatically pass __FILE__ and __LINE__
#define CHECK_CUDA(call) checkCudaResult(call, __FILE__, __LINE__)
#define CHECK_CUDNN(call) checkCudnnResult(call, __FILE__, __LINE__)

/**
 * @brief Calculates the linear index for a 4D tensor (NCHW layout).
 *
 * @param n Batch index.
 * @param c Channel index.
 * @param h Height index.
 * @param w Width index.
 * @param C Total number of channels.
 * @param H Tensor height.
 * @param W Tensor width.
 * @return The linear index in the flat array.
 */
__forceinline__ __device__ int get_nchw_index(int n, int c, int h, int w, int C, int H, int W) {
    // index = n * C*H*W + c * H*W + h * W + w
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

/**
 * @brief Calculates the linear index for a 4D filter tensor (typically OutC, InC, KernelH, KernelW layout).
 *
 * @param out_c Output channel index.
 * @param in_c Input channel index.
 * @param k_h Kernel height index.
 * @param k_w Kernel width index.
 * @param InC Total number of input channels.
 * @param KH Kernel height.
 * @param KW Kernel width.
 * @return The linear index in the flat filter array.
 */
__forceinline__ __device__ int get_filter_index(int out_c, int in_c, int k_h, int k_w, int InC, int KH, int KW) {
    // index = out_c * InC*KH*KW + in_c * KH*KW + k_h * KW + k_w
    return out_c * (InC * KH * KW) + in_c * (KH * KW) + k_h * KW + k_w;
}

/**
 * @brief Calculates the input tensor's height coordinate corresponding to an output coordinate and kernel position.
 * Handles stride and padding.
 *
 * @param out_h Output height coordinate.
 * @param k_h Kernel height coordinate.
 * @param stride_h Vertical stride.
 * @param pad_h Vertical padding.
 * @return The corresponding input height coordinate.
 */
__forceinline__ __device__ int get_input_h(int out_h, int k_h, int stride_h, int pad_h) {
    return out_h * stride_h + k_h - pad_h;
}

/**
 * @brief Calculates the input tensor's width coordinate corresponding to an output coordinate and kernel position.
 * Handles stride and padding.
 *
 * @param out_w Output width coordinate.
 * @param k_w Kernel width coordinate.
 * @param stride_w Horizontal stride.
 * @param pad_w Horizontal padding.
 * @return The corresponding input width coordinate.
 */
__forceinline__ __device__ int get_input_w(int out_w, int k_w, int stride_w, int pad_w) {
    return out_w * stride_w + k_w - pad_w;
}

/**
 * @brief Simple 2D convolution kernel for demonstration purposes.
 * 
 * This kernel performs a basic 2D convolution operation on the input tensor
 * with the provided filter. It handles the input and output dimensions,
 * strides, and padding.
 * 
 * It doesn't do tiling.
 * 
 * @param input Input tensor (N, C_in, H_in, W_in)
 * @param filter Filter tensor (C_out, C_in, KH, KW)
 * @param output Output tensor (N, C_out, H_out, W_out)
 * @param N Batch size
 * @param C_in Input channels,
 * @param H_in Input height,
 * @param W_in Input width,
 * @param C_out Output channels,
 * @param H_out Output height,
 * @param W_out Output width,
 * @param KH Kernel height,
 * @param KW Kernel width,
 * @param stride_h Vertical stride,
 * @param stride_w Horizontal stride,
 * @param pad_h Vertical padding,
 * @param pad_w Horizontal padding
 */
__global__ void simple_conv2d_kernel(
    const float* input,     // Input Tensor (N, C_in, H_in, W_in)
    const float* filter,    // Filter Tensor (C_out, C_in, KH, KW)
    float* output,          // Output Tensor (N, C_out, H_out, W_out)
    // --- Dimensions ---
    int N,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW,
    // --- Convolution Parameters ---
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = threadIdx.z;
    int n = blockIdx.z;

    // Out of bounds check
    if (w_out >= W_out || h_out >= H_out || c_out >= C_out || n >= N) {
        return;
    }

    float acc = 0.0f;

    for (int k_h = 0; k_h < KH; k_h++) {
        for (int k_w = 0; k_w < KW; k_w++) {
            int h_in = get_input_h(h_out, k_h, stride_h, pad_h);
            int w_in = get_input_w(w_out, k_w, stride_w, pad_w);

            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                for (int c_in = 0; c_in < C_in; c_in++) {
                    acc += input[get_nchw_index(n, c_in, h_in, w_in, C_in, H_in, W_in)] * shared_filter[c_in * KH * KW + k_h * KW + k_w];
                }
            }
        }
    }

    output[get_nchw_index(n, c_out, h_out, w_out, C_out, H_out, W_out)] = acc;
}

// Function to print a small section of a tensor for verification
void printTensorSection(const float* tensor, int n, int c, int h, int w, int C, int H, int W, const std::string& name) {
    std::cout << "=== " << name << " (batch=" << n << ", channel=" << c << ") ===" << std::endl;
    
    // Print a small window around the specified position
    int window_size = 5;
    int h_start = std::max(0, h - window_size/2);
    int h_end = std::min(H, h + window_size/2 + 1);
    int w_start = std::max(0, w - window_size/2);
    int w_end = std::min(W, w + window_size/2 + 1);
    
    for (int i = h_start; i < h_end; i++) {
        for (int j = w_start; j < w_end; j++) {
            int idx = n * (C * H * W) + c * (H * W) + i * W + j;
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << tensor[idx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to initialize a randomized symmetrical kernel
void initializeSymmetricalKernel(float* kernel, int C_out, int C_in, int KH, int KW) {
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    
    // For each output channel, create a different random symmetrical kernel
    for (int out_c = 0; out_c < C_out; out_c++) {
        // Create a random symmetrical pattern for this filter
        float kernel_pattern[3][3];
        
        // Fill only half of the kernel with random values
        for (int i = 0; i < KH; i++) {
            for (int j = 0; j <= i; j++) {
                float random_value = dist(gen);
                kernel_pattern[i][j] = random_value;
                // Mirror to maintain symmetry
                kernel_pattern[j][i] = random_value;
            }
        }
        
        // Calculate sum for normalization
        float sum = 0.0f;
        for (int i = 0; i < KH; i++) {
            for (int j = 0; j < KW; j++) {
                sum += kernel_pattern[i][j];
            }
        }
        
        // Apply the pattern to all input channels for this output channel
        for (int in_c = 0; in_c < C_in; in_c++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int idx = out_c * (C_in * KH * KW) + in_c * (KH * KW) + kh * KW + kw;
                    kernel[idx] = kernel_pattern[kh][kw] / sum;  // Normalize by sum of weights
                }
            }
        }
    }
}

// Function to perform convolution using cuDNN
void cudnnConv2d(
    const float* h_input, const float* h_filter, float* h_output_cudnn,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW, int stride_h, int stride_w, int pad_h, int pad_w
) {
    // Create cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    
    // Create tensor descriptors
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    
    // Set tensor descriptors
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C_in, H_in, W_in
    ));
    
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C_out, H_out, W_out
    ));
    
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        C_out, C_in, KH, KW
    ));
    
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor, pad_h, pad_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    ));
    
    // Allocate device memory
    size_t input_bytes = N * C_in * H_in * W_in * sizeof(float);
    size_t filter_bytes = C_out * C_in * KH * KW * sizeof(float);
    size_t output_bytes = N * C_out * H_out * W_out * sizeof(float);
    
    float *d_input, *d_filter, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    
    // Copy input and filter to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice));
    
    // Find best algorithm and workspace size using cudnnFindConvolutionForwardAlgorithm
    // Possible algorithms:
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    // CUDNN_CONVOLUTION_FWD_ALGO_GEMM
    // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
    // CUDNN_CONVOLUTION_FWD_ALGO_FFT
    // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int returnedAlgoCount = 1; // Request only the fastest algorithm
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_descriptor,
        filter_descriptor,
        convolution_descriptor,
        output_descriptor,
        1, // Request 1 algorithm result
        &returnedAlgoCount, // Number of algorithms returned
        &algoPerf // Pointer to the performance result struct
    ));

    // Check if an algorithm was found
    if (returnedAlgoCount == 0) {
        std::cerr << "cuDNN could not find a suitable convolution algorithm." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudnnConvolutionFwdAlgo_t algorithm = algoPerf.algo;
    workspace_bytes = algoPerf.memory; // Get workspace size from the perf result

    // Allocate workspace
    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
    }
    
    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    
    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Record start time
    CHECK_CUDA(cudaEventRecord(start));
    
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, d_input, filter_descriptor, d_filter,
        convolution_descriptor, algorithm, d_workspace, workspace_bytes,
        &beta, output_descriptor, d_output
    ));
    
    // Record end time
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuDNN convolution execution time: " << milliseconds << " ms" << std::endl;
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output, output_bytes, cudaMemcpyDeviceToHost));
    
    // Clean up
    if (d_workspace) {
        CHECK_CUDA(cudaFree(d_workspace));
    }
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_descriptor));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// Function to compare two tensors and compute the maximum absolute difference
float compareOutputs(const float* output1, const float* output2, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = std::abs(output1[i] - output2[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

int main(int argc, char** argv) {
    // Calculate output dimensions based on input, kernel, stride, and padding
    int H_out = (HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
    int W_out = (WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;
    
    // Output channels will be the same as input channels for simplicity
    int C_out = CHANNELS;
    
    // Calculate buffer sizes
    size_t input_bytes = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH * sizeof(float);
    size_t filter_bytes = C_out * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(float);
    size_t output_bytes = BATCH_SIZE * C_out * H_out * W_out * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_filter, *h_output, *h_output_cudnn;
    CHECK_CUDA(cudaMallocHost(&h_input, input_bytes));
    CHECK_CUDA(cudaMallocHost(&h_filter, filter_bytes));
    CHECK_CUDA(cudaMallocHost(&h_output, output_bytes));
    CHECK_CUDA(cudaMallocHost(&h_output_cudnn, output_bytes));
    
    // Initialize input with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < BATCH_SIZE * CHANNELS * HEIGHT * WIDTH; i++) {
        h_input[i] = dist(gen);
    }
    
    // Initialize filter with symmetrical [1 2 1] pattern
    initializeSymmetricalKernel(h_filter, C_out, CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH);
    
    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice));
    
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16, OUTPUT_CHANNELS);  // 16x16 threads per block, 1 output channel per thread
    dim3 gridDim(
        (W_out + blockDim.x - 1) / blockDim.x,
        (H_out + blockDim.y - 1) / blockDim.y,
        BATCH_SIZE  // One batch per grid z-dimension
    );
    
    // Print kernel information
    std::cout << "Kernel information:" << std::endl;
    std::cout << "  Kernel shape: [" << KERNEL_HEIGHT << "x" << KERNEL_WIDTH << "]" << std::endl;
    std::cout << "  Kernel pattern: [1 2 1] in both directions" << std::endl;
    std::cout << "  Input shape: [" << BATCH_SIZE << ", " << CHANNELS << ", " << HEIGHT << ", " << WIDTH << "]" << std::endl;
    std::cout << "  Output shape: [" << BATCH_SIZE << ", " << C_out << ", " << H_out << ", " << W_out << "]" << std::endl;
    std::cout << std::endl;
    
    // Launch kernel
    std::cout << "Launching kernel with grid: [" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << "]" << std::endl;
    std::cout << "Block dimensions: [" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << "]" << std::endl;
    
    // Record start time
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    // Launch the convolution kernel
    simple_conv2d_kernel<<<gridDim, blockDim>>>(
        d_input, d_filter, d_output,
        BATCH_SIZE,
        CHANNELS, HEIGHT, WIDTH,
        C_out, H_out, W_out,
        KERNEL_HEIGHT, KERNEL_WIDTH,
        STRIDE, STRIDE,
        PADDING, PADDING
    );
    
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    
    // Record end time and calculate elapsed time
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Custom kernel execution time: " << milliseconds << " ms" << std::endl;
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    
    // Run cuDNN convolution for verification
    std::cout << "\nRunning cuDNN convolution for verification..." << std::endl;
    cudnnConv2d(
        h_input, h_filter, h_output_cudnn,
        BATCH_SIZE, CHANNELS, HEIGHT, WIDTH,
        C_out, H_out, W_out,
        KERNEL_HEIGHT, KERNEL_WIDTH,
        STRIDE, STRIDE, PADDING, PADDING
    );
    
    // Compare outputs
    int output_size = BATCH_SIZE * C_out * H_out * W_out;
    float max_diff = compareOutputs(h_output, h_output_cudnn, output_size);
    std::cout << "Maximum absolute difference between custom and cuDNN: " << max_diff << std::endl;
    
    // Print a small section of the input and output for verification
    int sample_batch = 0;
    int sample_channel = 0;
    int sample_h = H_out / 2;
    int sample_w = W_out / 2;
    
    printTensorSection(h_input, sample_batch, sample_channel, sample_h, sample_w, CHANNELS, HEIGHT, WIDTH, "Input Tensor");
    printTensorSection(h_output, sample_batch, sample_channel, sample_h, sample_w, C_out, H_out, W_out, "Custom Output Tensor");
    printTensorSection(h_output_cudnn, sample_batch, sample_channel, sample_h, sample_w, C_out, H_out, W_out, "cuDNN Output Tensor");
    
    // Clean up
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_filter));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFreeHost(h_output_cudnn));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
