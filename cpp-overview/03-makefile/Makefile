.PHONY: 01 01_obj 01_obj_exe_run clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_90

01:
	@$(GCC) -o 01 01.c

# just compiles to object file
01_obj:
	@$(GCC) -c 01.c -o 01.o

# compiles and runs the object file (ensure 01_obj is up to
# date by running 01_obj first if it hasn't been run yet)
01_obj_exe_run: 01_obj
	@$(GCC) 01.o -o 01
	@./01

02:
	@$(NVCC) $(CUDA_FLAGS) -o 02 02.cu

clean:
	rm 01 01.o 02

