import argparse
import subprocess
import re
import os
import itertools
import tempfile
import shutil
import multiprocessing
from typing import List, Tuple, Optional, Dict

# --- Configuration ---
NVCC_PATH = "nvcc"  # Or specify the full path if not in PATH
NVCC_FLAGS = ["-O3", "-arch=sm_90"]
EXECUTABLE_NAME = "temp_matmul_exec"

# Parameter ranges (adjust as needed)
BM_RANGE = [64, 128, 256]
BN_RANGE = [64, 128, 256]
BK_RANGE = [8, 16, 32, 64]  # Must be multiple of 4
TM_RANGE = [4, 8]
TN_RANGE = [4, 8]        # Must be multiple of 4

MAX_THREADS_PER_BLOCK = 2048
# Max shared memory per block in bytes (e.g., 48 KB for many GPUs)
# Check your GPU's specs (deviceQuery) if unsure. Can be up to 96KB or more.
MAX_SHARED_MEMORY_BYTES = 220 * 1024
# --- End Configuration ---

def modify_source(original_content, bm, bn, bk, tm, tn):
    """Modifies the constexpr definitions in the source code string."""
    content = original_content
    content = re.sub(r"constexpr int BM\s*=\s*\d+;", f"constexpr int BM = {bm};", content)
    content = re.sub(r"constexpr int BN\s*=\s*\d+;", f"constexpr int BN = {bn};", content)
    content = re.sub(r"constexpr int BK\s*=\s*\d+;", f"constexpr int BK = {bk};", content)
    content = re.sub(r"constexpr int TM\s*=\s*\d+;", f"constexpr int TM = {tm};", content)
    content = re.sub(r"constexpr int TN\s*=\s*\d+;", f"constexpr int TN = {tn};", content)
    content = re.sub(r"\\brun_benchmark\\b", "run_benchmark_gpu_only", content)
    return content

def parse_gflops(output):
    """Extracts GFLOPS value from the program's output."""
    match = re.search(r"GPU Performance:\s*([\d\.]+)\s*GFLOPS", output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def check_constraints(bm, bn, bk, tm, tn):
    """Checks if the parameter combination is valid."""
    threads_per_block = bm * bn / (tm * tn)
    if threads_per_block == 0: # Avoid division by zero if tm/tn are large relative to bm/bn
        # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): Invalid thread configuration leads to zero threads.")
        return False
    if threads_per_block > MAX_THREADS_PER_BLOCK:
        # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): Threads per block ({threads_per_block}) exceeds limit ({MAX_THREADS_PER_BLOCK})")
        return False

    shared_mem_needed = (bm * bk + bk * bn) * 8 # sizeof(float)
    if shared_mem_needed > MAX_SHARED_MEMORY_BYTES:
        # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): Shared memory ({shared_mem_needed} bytes) exceeds limit ({MAX_SHARED_MEMORY_BYTES})")
        return False

    # Register pressure constraint (assuming 64k 32-bit registers per SM)
    # Formula provided: regs_per_thread = TM * TN + TM + TN + 16
    # Total regs = regs_per_thread * threads_per_block
    MAX_REGISTERS_PER_SM = 64 * 1024
    regs_per_thread_approx = tm * tn + tm + tn + 13
    total_regs_approx = regs_per_thread_approx * threads_per_block
    if total_regs_approx > MAX_REGISTERS_PER_SM:
        # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): Estimated register usage ({total_regs_approx}) exceeds limit ({MAX_REGISTERS_PER_SM})")
        return False

    if bk % 4 != 0:
         # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): BK must be a multiple of 4")
         return False

    if tn % 4 != 0:
         # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): TN must be a multiple of 4")
         return False

    # Constraint: BN * 4 <= BK * TM * TN && BM * 4 <= BK * TM * TN
    if bn * 4 > bk * tm * tn or bm * 4 > bk * tm * tn:
        # print(f"Skipping ({bm},{bn},{bk},{tm},{tn}): BN*4 must be <= BK*TM*TN and BM*4 must be <= BK*TM*TN")
        return False

    # blockDim.x = bm * bn / (tm * tn) needs to be divisible by (bn / tn)
    # This simplifies to bm / tm, which is implicitly handled if bm, tm are powers of 2? Let's assume it's fine for now.
    # Re-check if compilation errors occur related to indexing.

    return True

def run_tuning_worker(
    gpu_id: int,
    param_chunk: List[Tuple[int, int, int, int, int]],
    original_content: str,
    source_file_basename: str,
    header_files: Optional[List[str]],
    nvcc_path: str,
    nvcc_flags: List[str],
    executable_name_base: str
) -> Tuple[Optional[Tuple[int, int, int, int, int]], float]:
    """
    Worker function to run autotuning for a subset of parameters on a specific GPU.

    Returns:
        A tuple containing the best parameters found by this worker and the corresponding GFLOPS.
        Returns (None, 0.0) if no successful runs occurred for this worker.
    """
    worker_best_gflops = 0.0
    worker_best_params = None
    pid = os.getpid()
    print(f"[GPU {gpu_id}, PID {pid}] Starting worker for {len(param_chunk)} combinations.")

    # Set CUDA_VISIBLE_DEVICES for this worker process and its children
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create a unique temporary directory for this worker
    temp_dir = tempfile.mkdtemp(prefix=f"matmul_autotune_gpu{gpu_id}_pid{pid}_")
    temp_source_path = os.path.join(temp_dir, source_file_basename)
    # Make executable name unique per worker to avoid potential clashes if temp dirs aren't isolated perfectly
    temp_executable_path = os.path.join(temp_dir, f"{executable_name_base}_gpu{gpu_id}")

    # Copy header files to temp directory if specified
    if header_files:
        for header_file in header_files:
            if os.path.exists(header_file):
                try:
                    shutil.copy(header_file, temp_dir)
                except Exception as e:
                    print(f"[GPU {gpu_id}, PID {pid}] Warning: Failed to copy header {header_file}: {e}")
            # else: # Warning printed in main process already
            #     print(f"[GPU {gpu_id}, PID {pid}] Warning: Header file not found: {header_file}")

    try:
        for params in param_chunk:
            bm, bn, bk, tm, tn = params

            # Constraints already checked in the main process, but double-check won't hurt
            # if not check_constraints(bm, bn, bk, tm, tn):
            #     continue

            # print(f"[GPU {gpu_id}, PID {pid}] Testing BM={bm}, BN={bn}, BK={bk}, TM={tm}, TN={tn}...") # Verbose

            modified_content = modify_source(original_content, bm, bn, bk, tm, tn)

            try:
                with open(temp_source_path, 'w') as f:
                    f.write(modified_content)
            except IOError as e:
                print(f"[GPU {gpu_id}, PID {pid}] Error writing temporary source file: {e}. Skipping {params}.")
                continue

            # Compile
            compile_cmd = [nvcc_path] + nvcc_flags + ["-o", temp_executable_path, temp_source_path]
            try:
                # Use the modified environment for subprocesses
                compile_result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True, timeout=60, env=env)
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}, PID {pid}] Compilation failed for {params}:")
                print(e.stderr)
                continue
            except subprocess.TimeoutExpired:
                print(f"[GPU {gpu_id}, PID {pid}] Compilation timed out for {params}. Skipping.")
                continue

            # Run with retries
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    # Use the modified environment for subprocesses
                    run_result = subprocess.run([temp_executable_path], check=True, capture_output=True, text=True, timeout=120, env=env)
                    output = run_result.stdout
                    current_gflops = parse_gflops(output)
                    success = True # Mark as success if run completes and parsing is attempted

                    if current_gflops is not None:
                        # print(f"[GPU {gpu_id}, PID {pid}] Params {params} Achieved: {current_gflops:.2f} GFLOPS") # Verbose
                        if current_gflops > worker_best_gflops:
                            worker_best_gflops = current_gflops
                            worker_best_params = params
                            print(f"[GPU {gpu_id}, PID {pid}] *** New best for this worker: {params} -> {current_gflops:.2f} GFLOPS ***")
                    # else: # Error parsing (don't retry for parsing errors)
                        # print(f"[GPU {gpu_id}, PID {pid}] Could not parse GFLOPS for {params}. Output:\n{output}")
                    break # Exit retry loop on success

                except subprocess.CalledProcessError as e:
                    print(f"[GPU {gpu_id}, PID {pid}] Execution failed for {params} (Attempt {attempt + 1}/{max_retries}):")
                    print(f"  Return code: {e.returncode}")
                    # print(f"  Stderr:\n{e.stderr}") # Optionally print stderr on failure
                    if attempt + 1 == max_retries:
                         print(f"  Giving up after {max_retries} attempts.")
                    else:
                        print("  Retrying...")
                        # Optional: time.sleep(1) # Add a small delay before retry
                except subprocess.TimeoutExpired:
                    print(f"[GPU {gpu_id}, PID {pid}] Execution timed out for {params} (Attempt {attempt + 1}/{max_retries}).")
                    if attempt + 1 == max_retries:
                         print(f"  Giving up after {max_retries} attempts.")
                    else:
                        print("  Retrying...")
                        # Optional: time.sleep(1)
                    # No break here, loop continues for retry

            # Clean up executable only after all retries for a parameter set are done
            if os.path.exists(temp_executable_path):
                try:
                    os.remove(temp_executable_path)
                except OSError as e:
                     print(f"[GPU {gpu_id}, PID {pid}] Warning: could not remove temp executable {temp_executable_path}: {e}")

            # If all retries failed, continue to the next parameter combination
            if not success:
                continue # Skip to the next params in the loop

    finally:
        # Clean up temporary directory
        # print(f"[GPU {gpu_id}, PID {pid}] Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"[GPU {gpu_id}, PID {pid}] Warning: could not remove temp directory {temp_dir}: {e}")

    print(f"[GPU {gpu_id}, PID {pid}] Worker finished. Best GFLOPS: {worker_best_gflops:.2f}")
    return worker_best_params, worker_best_gflops


def main(source_file, header_files=None, num_gpus=1):
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return

    try:
        with open(source_file, 'r') as f:
            original_content = f.read()
    except IOError as e:
        print(f"Error reading source file {source_file}: {e}")
        return

    best_gflops = 0.0
    best_params = None
    tested_count = 0
    valid_count = 0
    total_combinations = 0 # Will calculate after filtering

    # Filter combinations first
    print("Checking parameter constraints...")
    all_param_combinations = list(itertools.product(BM_RANGE, BN_RANGE, BK_RANGE, TM_RANGE, TN_RANGE))
    valid_param_combinations = [
        params for params in all_param_combinations if check_constraints(*params)
    ]
    total_combinations = len(all_param_combinations)
    valid_count = len(valid_param_combinations)

    print(f"Starting autotuning for {source_file} using {num_gpus} GPU(s)")
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Valid combinations meeting constraints: {valid_count}")
    if valid_count == 0:
        print("No valid parameter combinations found. Exiting.")
        return
    print(f"NVCC path: {NVCC_PATH}")
    print(f"NVCC flags: {' '.join(NVCC_FLAGS)}")
    if header_files:
        # Check header existence once in the main process
        found_headers = []
        for header_file in header_files:
            if os.path.exists(header_file):
                found_headers.append(header_file)
            else:
                 print(f"  Warning: Header file not found, will not be copied by workers: {header_file}")
        if found_headers:
            print(f"Including header files: {', '.join(found_headers)}")
        header_files = found_headers # Pass only existing headers to workers

    print("-" * 30)

    # Split valid combinations among workers
    chunks = [[] for _ in range(num_gpus)]
    for i, params in enumerate(valid_param_combinations):
        chunks[i % num_gpus].append(params)

    # Prepare arguments for each worker
    worker_args = []
    source_file_basename = os.path.basename(source_file)
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]: # Skip GPUs with no work
            print(f"No work assigned to GPU {gpu_id}, skipping worker.")
            continue
        worker_args.append((
            gpu_id,
            chunks[gpu_id],
            original_content,
            source_file_basename,
            header_files,
            NVCC_PATH,
            NVCC_FLAGS,
            EXECUTABLE_NAME
        ))

    if not worker_args:
        print("No work to distribute. Exiting.")
        return

    # Run workers in parallel
    print(f"Launching {len(worker_args)} worker processes...")
    with multiprocessing.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(run_tuning_worker, worker_args)

    print("-" * 30)
    print("All workers finished. Aggregating results...")

    # Aggregate results
    total_tested_in_workers = 0 # This count might be less precise now
    for result_params, result_gflops in results:
        # tested_count needs rethink if we want accurate count across workers
        if result_gflops > best_gflops:
            best_gflops = result_gflops
            best_params = result_params

    print("-" * 30)
    print(f"Autotuning finished.")
    # print(f"Checked {valid_count}/{total_combinations} valid combinations.") # Valid count is accurate
    # print(f"Successfully compiled and ran {tested_count} combinations.") # tested_count is hard to track accurately now

    if best_params:
        bm, bn, bk, tm, tn = best_params
        print("\nBest parameters found:")
        print(f"  BM = {bm}")
        print(f"  BN = {bn}")
        print(f"  BK = {bk}")
        print(f"  TM = {tm}")
        print(f"  TN = {tn}")
        print(f"Best GPU Performance: {best_gflops:.2f} GFLOPS")
    else:
        print("\nNo successful runs completed or GFLOPS could not be parsed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotune MatMul CUDA kernel parameters.")
    parser.add_argument("source_file", help="Path to the CUDA source file (.cu)")
    parser.add_argument("--headers", nargs="+", help="Header files to copy to the temporary directory")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for parallel tuning (default: 1)")
    args = parser.parse_args()
    main(args.source_file, args.headers, args.gpus)