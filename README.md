# Learning CUDA – Notes & Hands-On Examples

This repository contains a **self-contained, incremental tour of NVIDIA CUDA and GPU programming**
— from a refresher on C/C++ all the way to hand-tuned matrix-multiplication kernels.
It grew out of personal study notes that were kept in a handful of Org-mode
files and a collection of small, focused code snippets.  Everything is now
consolidated here so that the notes render nicely on GitHub and the code can be
compiled or copy-pasted straight into your projects.

The material is **not a full course** and it does not try to compete with the
excellent official CUDA documentation.  Instead, think of it as a curated
check-list of the things you usually need when you start writing your own
kernels:

* where GPU hardware shines compared to CPUs,
* the basic programming model (thread / warp / block / grid),
* how to compile and profile simple kernels,
* the high-level CUDA libraries you should know (cuBLAS, cuDNN, NCCL, …)
* and finally a deep dive into **fast GEMM / SGEMM** implementations.

---

## Table of contents

| Section | What you will find | Source file |
|---------|-------------------|-------------|
| 1. Introduction | A high-level overview of today’s deep-learning software stack and where CUDA fits in | `001-intro.org` |
| 2. Setup | One-liner instructions for getting a CUDA-capable environment up and running (spoiler: use the official Docker images) | `002-setup.org` |
| 3. C/C++ Refresher | How to compile C, C++ *and* CUDA, plus a primer on the C pre-processor | `003-cpp-overview.org` |
| 4. GPU Basics | CPU vs GPU architecture, a brief hardware history and the vocabulary every CUDA programmer must know | `004-intro-gpu.org` |
| 5. Your First Kernel | A worked example that walks from `deviceQuery` to a hand-written “Hello GPU” kernel | `005-first-kernel.org` |
| 6. CUDA Libraries | cuBLAS(Lt/X/Dx), cuDNN, NCCL, MIG and more—including handy error-checking macros | `006-cuda-api.org` |
| 7. Faster MatMul | From a naïve O(N³) kernel to register / shared-memory / tensor-core madness, heavily inspired by the fantastic write-up from Siboehm (Anthropic) | `007-faster-matmul.org` |

The original Org files are kept verbatim for people who prefer Org-mode.  You
can open them with Emacs or view them on GitHub directly.

---

## Code layout

The repository pairs each note with one or more minimal, buildable examples:

```
cpp-overview/           # -> C / C++ examples that also run under nvcc
writing-first-kernels/  # -> Kernels that accompany section 5
cuda-api/               # -> cuBLAS / cuDNN samples and Makefiles
faster-matmul/          # -> Every optimisation step discussed in section 7
```

Most directories have a `Makefile` or a one-liner comment that shows how to
compile the code, e.g.

```bash
# compile a plain CUDA file
nvcc -arch=sm_90 -o main main.cu

# run nvcc in “pass-through” mode so that host code is built by g++
nvcc -x cu -Xcompiler="-O3 -Wall" -arch=sm_90 *.cu -o example
```

> **Tip:** Always pass `--generate-line-info` when you plan to profile with
> `nsight-compute` (`ncu`).  Source-line mapping in the GUI is priceless.

---

## Prerequisites

* CUDA 12 or newer (all samples were tested with CUDA 12.3).
* A GPU with compute-capability 8.0 or later will run everything, but most code
  will happily compile for earlier SM versions as well (drop `-arch=sm_90`).
* CMake is **not** required; plain `nvcc` / `make` is used throughout to keep
  the focus on the kernels.

If you don’t want to install the toolkit locally, spin up the official CUDA
Docker image and mount this repo:

```bash
docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04
```

---

## Referenced material

* NVIDIA CUDA Toolkit documentation: https://docs.nvidia.com/cuda/
* Siboehm’s blog post “How to Write Fast Matrix Multiply”: https://siboehm.com/articles/22/CUDA-MMM

Images in the `assets/` folder are re-used from the CUDA docs and public blog
posts for educational purposes.  All other code and text in this repository is
released under the MIT License (see `LICENSE`).

---

## Contributing

This is a learning project first and foremost.  If you spot an inaccuracy,
spelling mistake or have a performance tweak to share, feel very welcome to
open a pull request or an issue.

Happy GPU hacking! 🚀
