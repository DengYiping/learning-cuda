#include <stdio.h>
#include "cuda_runtime.h"

int main() {
    int device_id = 0; // Or the ID you expect the H100 to be
    cudaDeviceProp properties;
    cudaError_t prop_err = cudaGetDeviceProperties(&properties, device_id);
    if (prop_err != cudaSuccess) {
         fprintf(stderr, "Failed to get properties for device %d: %s\n", device_id, cudaGetErrorString(prop_err));
         return 1;
    }
    printf("Device %d Name: %s\n", device_id, properties.name);
    printf("Compute Capability: %d.%d\n", properties.major, properties.minor);


    int available_shared_memory = 0;
    cudaError_t attr_err = cudaDeviceGetAttribute(&available_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
     if (attr_err != cudaSuccess) {
         fprintf(stderr, "Failed to get attribute for device %d: %s\n", device_id, cudaGetErrorString(attr_err));
         return 1;
    }

    printf("cudaDevAttrMaxSharedMemoryPerBlock for device %d: %d bytes (%d KiB)\n",
           device_id, available_shared_memory, available_shared_memory / 1024);

    return 0;
}