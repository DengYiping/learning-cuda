#include <iostream>       // For std::cerr
#include <string>         // For std::string
#include <cstdlib>        // For exit, EXIT_FAILURE
#include <cuda_runtime.h> // For CUDA types and functions
#include <cudnn.h>        // For cuDNN types and functions
#include <iomanip>        // For std::setw, std::setprecision

// Define dimensions for CNN
constexpr int BATCH_SIZE = 256;
constexpr int CHANNELS = 32;
constexpr int HEIGHT = 224;
constexpr int WIDTH = 224;
constexpr int TOTAL_ELEMENTS = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH;

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

// 1d grid, 1d kernel
__global__ void naive_tanh(float* d_A, float* d_B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_B[idx] = tanhf(d_A[idx]);
}

// Function to print results in a formatted way
void printResults(const float* input, const float* output, int count) {
    std::cout << "=== Results (first 10 elements) ===" << std::endl;
    std::cout << std::setw(5) << "Index" << std::setw(15) << "Input" << std::setw(15) << "Output (tanh)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Print first 10 elements or fewer if count < 10
    int display_count = std::min(10, count);
    for (int i = 0; i < display_count; i++) {
        std::cout << std::setw(5) << i
                  << std::setw(15) << std::fixed << std::setprecision(6) << input[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << output[i]
                  << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
  // Prep
  constexpr auto BUFFER_SIZE = sizeof(float) * TOTAL_ELEMENTS;
  float *h_input, *h_output, *d_input, *d_output;
  CHECK_CUDA(cudaMallocHost(&h_input, BUFFER_SIZE));
  CHECK_CUDA(cudaMallocHost(&h_output, BUFFER_SIZE));
  CHECK_CUDA(cudaMalloc(&d_input, BUFFER_SIZE));
  CHECK_CUDA(cudaMalloc(&d_output, BUFFER_SIZE));

  for (int i = 0; i < TOTAL_ELEMENTS; i++) {
    h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random values between -1 and 1
  }
  CHECK_CUDA(cudaMemcpy(d_input, h_input, BUFFER_SIZE, cudaMemcpyHostToDevice));

  // Naive tanh implementation
  {
    // cuda stream timing
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    naive_tanh<<<TOTAL_ELEMENTS / 256, 256, 0, stream>>>(d_input, d_output);
    CHECK_CUDA(cudaEventRecord(stop, stream));

    CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, BUFFER_SIZE, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Print results
    std::cout << "Naive tanh results:" << std::endl;
    printResults(h_input, h_output, TOTAL_ELEMENTS);

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Naive tanh time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
  }

  // cuDNN tanh implementation
  {
    // cuda stream timing
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // TODO: cuDNN tanh implementation
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));

    // Set input descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH));

    // Set output descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH));

    // Create activation descriptor
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0f));

    // Define scaling factors for activation
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < 100; i++) {
      CHECK_CUDNN(cudnnActivationForward(handle, activationDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));
    }

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Run cuDNN tanh (workspace is deprecated and not needed)
    CHECK_CUDNN(cudnnActivationForward(handle, activationDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // Copy output to host
    CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, BUFFER_SIZE, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Print results
    std::cout << "cuDNN tanh results:" << std::endl;
    printResults(h_input, h_output, TOTAL_ELEMENTS);

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuDNN tanh time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroy(handle));
  }

  // Cleanup memory
  CHECK_CUDA(cudaFreeHost(h_input));
  CHECK_CUDA(cudaFreeHost(h_output));
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
}
