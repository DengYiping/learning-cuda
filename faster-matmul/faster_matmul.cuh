#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>       // Add this for memset
#include <time.h>         // For benchmarking
#include <iomanip>        // For std::setw, std::setprecision
#include <random>         // For random number generation
#include <cmath>          // For std::abs
#include <chrono>         // For CPU timing

#define CHECK_CUDA(call) checkCudaResult(call, __FILE__, __LINE__)

void checkCudaResult(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << file << "' in line " << line << " : "
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cpu_matmul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void check_result_eq(float* ground_truth, float* result, int size) {
    for (int i = 0; i < size; i++) {
        if (std::abs(ground_truth[i] - result[i]) > 1e-6) {
        }
    }
}

// Function to verify results between CPU and GPU implementations
bool verify_results(float* cpu_result, float* gpu_result, int size, float tolerance = 1e-3) {
    bool pass = true;
    for (int i = 0; i < size; i++) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > tolerance) {
            std::cout << "Result mismatch at index " << i << ": "
                      << gpu_result[i] << " vs " << cpu_result[i] << std::endl;
            pass = false;
            break;
        }
    }
    
    if (pass) {
        std::cout << "Results match between CPU and GPU!" << std::endl;
    } else {
        std::cout << "Results DO NOT match between CPU and GPU!" << std::endl;
    }
    
    return pass;
}

// Function type for kernel launchers
template<typename T>
using kernel_launcher_t = void (*)(T* d_A, T* d_B, T* d_C, int m, int n, int k, cudaStream_t stream);

// Type conversion function types
template<typename T>
using to_type_converter_t = T (*)(float);

template<typename T>
using from_type_converter_t = float (*)(T);

// Default converters
template<typename T>
T default_to_type(float val) {
    return static_cast<T>(val);
}

template<typename T>
float default_from_type(T val) {
    return static_cast<float>(val);
}

// Specialization for float type (no conversion needed)
template<>
float default_to_type<float>(float val) {
    return val;
}

template<>
float default_from_type<float>(float val) {
    return val;
}

// Run benchmark function that tests a kernel against CPU implementation
template<typename T>
float run_benchmark(
    kernel_launcher_t<T> launch_gpu_kernel, 
    int m, int n, int k, 
    int iterations = 10,
    to_type_converter_t<T> to_type = default_to_type<T>,
    from_type_converter_t<T> from_type = default_from_type<T>
) {
    size_t size_A = m * k * sizeof(T);
    size_t size_B = k * n * sizeof(T);
    size_t size_C = m * n * sizeof(T);
    
    // Allocate pinned host memory using cudaMallocHost
    T *h_A, *h_B, *h_C;
    float *h_C_cpu;
    CHECK_CUDA(cudaMallocHost((void**)&h_A, size_A));
    CHECK_CUDA(cudaMallocHost((void**)&h_B, size_B));
    CHECK_CUDA(cudaMallocHost((void**)&h_C, size_C));
    CHECK_CUDA(cudaMallocHost((void**)&h_C_cpu, m * n * sizeof(float)));  // Always use fp32 for CPU
    
    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < m * k; i++) {
        h_A[i] = to_type(dist(gen));
    }
    
    for (int i = 0; i < k * n; i++) {
        h_B[i] = to_type(dist(gen));
    }
    
    // Create float versions for CPU computation using pinned memory
    float *h_A_float, *h_B_float;
    CHECK_CUDA(cudaMallocHost((void**)&h_A_float, m * k * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_B_float, k * n * sizeof(float)));
    
    for (int i = 0; i < m * k; i++) {
        h_A_float[i] = from_type(h_A[i]);
    }
    
    for (int i = 0; i < k * n; i++) {
        h_B_float[i] = from_type(h_B[i]);
    }
    
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_C));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Compute reference result on CPU (always using fp32) with timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_A_float, h_B_float, h_C_cpu, m, n, k);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    
    std::cout << "CPU matrix multiplication time: " << cpu_duration.count() << " ms" << std::endl;
    
    // Calculate CPU performance
    double cpu_flops = 2.0 * m * n * k;  // multiply-add is 2 FLOP
    double cpu_gflops = (cpu_flops * 1e-9) / (cpu_duration.count() * 1e-3);
    std::cout << "CPU Performance: " << cpu_gflops << " GFLOPS" << std::endl;
    
    // Warmup GPU
    launch_gpu_kernel(d_A, d_B, d_C, m, n, k, stream);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark GPU implementation
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    
    for (int i = 0; i < iterations; i++) {
        launch_gpu_kernel(d_A, d_B, d_C, m, n, k, stream);
    }
    
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / iterations;
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Convert GPU results to float for verification using pinned memory
    float *h_C_float;
    CHECK_CUDA(cudaMallocHost((void**)&h_C_float, m * n * sizeof(float)));
    for (int i = 0; i < m * n; i++) {
        h_C_float[i] = from_type(h_C[i]);
    }
    
    // Verify results using the separate function
    verify_results(h_C_cpu, h_C_float, m * n);
    
    // Print a 3x3 section from both CPU and GPU results for comparison
    std::cout << "\nCPU result (3x3 section):" << std::endl;
    for (int i = 0; i < 3 && i < m; i++) {
        for (int j = 0; j < 3 && j < n; j++) {
            std::cout << std::fixed << std::setprecision(3) << h_C_cpu[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nGPU result (3x3 section):" << std::endl;
    for (int i = 0; i < 3 && i < m; i++) {
        for (int j = 0; j < 3 && j < n; j++) {
            std::cout << std::fixed << std::setprecision(3) << from_type(h_C[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }

    // Calculate and return performance
    double flops = 2.0 * m * n * k * iterations;  // multiply-add is 2 FLOP
    double gflops = (flops * 1e-9) / (milliseconds * 1e-3);
    
    std::cout << "Matrix dimensions: " << m << "x" << k << " * " << k << "x" << n << std::endl;
    std::cout << "Average kernel execution time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "GPU Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "GPU theoretical performance on H100: 67 teraFLOPS" << std::endl;
    std::cout << "Performance vs theoretical max in percentage: " << gflops / 67e3 * 100 << "%" << std::endl;
    std::cout << "GPU speedup over CPU: " << cpu_duration.count() / avg_time_ms << "x" << std::endl;

    // Clean up - use cudaFreeHost for pinned memory
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
    CHECK_CUDA(cudaFreeHost(h_C_cpu));
    CHECK_CUDA(cudaFreeHost(h_A_float));
    CHECK_CUDA(cudaFreeHost(h_B_float));
    CHECK_CUDA(cudaFreeHost(h_C_float));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return avg_time_ms;
}
