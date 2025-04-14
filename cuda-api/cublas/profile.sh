#!/bin/bash

# Ensure executables are built first
make

# Create directory for profiling results if it doesn't exist
mkdir -p profile_results

echo "Profiling gemm executable..."
ncu --set full --export profile_results/gemm_profile \
    --force-overwrite \
    ./gemm

echo "Profiling gemm_cublaslt executable..."
ncu --set full --export profile_results/gemm_cublaslt_profile \
    --force-overwrite \
    ./gemm_cublaslt

echo "Profiling complete. Results saved in profile_results/ directory." 