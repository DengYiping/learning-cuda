#!/usr/bin/env python3

import argparse
import mmap
import os
import re
import subprocess
import tempfile
import shutil

# --- Configuration ---
# Try to find CUDA_HOME automatically, otherwise rely on environment variable
CUDA_HOME_ENV = os.getenv('CUDA_HOME')
if CUDA_HOME_ENV and os.path.exists(os.path.join(CUDA_HOME_ENV, 'bin', 'nvcc')):
    CUDA_HOME = CUDA_HOME_ENV
else:
    # Try finding nvcc in PATH
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        # Infer CUDA_HOME from nvcc path (e.g., /usr/local/cuda/bin/nvcc -> /usr/local/cuda)
        CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))
        print(f"Inferred CUDA_HOME='{CUDA_HOME}' from nvcc path.")
    else:
        CUDA_HOME = None

# Check if CUDA_HOME is valid
if not CUDA_HOME or not os.path.exists(os.path.join(CUDA_HOME, 'bin', 'nvcc')):
    raise EnvironmentError(
        "CUDA_HOME environment variable is not set or nvcc not found. "
        "Please set CUDA_HOME to your CUDA installation directory (e.g., /usr/local/cuda)."
    )

NVCC_PATH = os.path.join(CUDA_HOME, 'bin', 'nvcc')
CUOBJDUMP_PATH = os.path.join(CUDA_HOME, 'bin', 'cuobjdump')

# Check if tools exist
if not os.path.exists(NVCC_PATH):
     raise FileNotFoundError(f"nvcc not found at {NVCC_PATH}")
if not os.path.exists(CUOBJDUMP_PATH):
     raise FileNotFoundError(f"cuobjdump not found at {CUOBJDUMP_PATH}")

# --- Helper Functions ---

def run_command(command, description):
    """Runs a command and checks for errors."""
    # print(f"Running: {' '.join(command)}") # Optional: verbose command output
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if result.stderr:
            print(f"STDERR ({description}):\n{result.stderr}")
        return result.stdout
    except FileNotFoundError as e:
        print(f"Error: Command not found - {e.filename}")
        print(f"Ensure CUDA binaries are in your PATH or CUDA_HOME is set correctly.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

def run_cuobjdump(file_path):
    """Runs cuobjdump to get SASS."""
    command = [CUOBJDUMP_PATH, '-sass', file_path]
    # Use check=False and handle error explicitly like in original optimize script
    # print(f"Running: {' '.join(command)}") # Optional: verbose command output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running cuobjdump:")
        print(f"Command: {' '.join(command)}")
        print(f"Return Code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    return result.stdout

def extract_hex_from_line(line):
    """Extracts the hex value from a SASS comment like /* 0x... */"""
    match = re.search(r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/', line)
    if not match:
        raise ValueError(f"Could not find hex pattern /* 0x... */ in line: {line}")
    return int(match.group(1), 16)

# --- Core Logic ---

def print_ffma_flags(cu_file, arch):
    """
    Compiles a .cu file, runs cuobjdump, parses SASS, and prints FFMA
    instructions with their reuse and yield flags.
    """
    print(f"Analyzing {cu_file} for arch {arch}")
    base_name = os.path.splitext(os.path.basename(cu_file))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        object_file = os.path.join(tmpdir, f"{base_name}.o")

        # 1. Compile .cu to .o
        print(f"Compiling {cu_file} to {object_file}...")
        compile_command = [
            NVCC_PATH,
            '-gencode', f'arch=compute_{arch.split("_")[1]},code=sm_{arch.split("_")[1]}',
            '-c', cu_file,
            '-o', object_file,
            # No specific optimization flags needed, defaults should suffice for SASS generation
        ]
        try:
            run_command(compile_command, "NVCC Compile")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
             print(f"Compilation failed: {e}")
             return # Cannot proceed

        # 2. Run cuobjdump on the object file
        print(f"Running cuobjdump on {object_file}...")
        try:
            sass_output = run_cuobjdump(object_file)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"cuobjdump failed: {e}")
            return # Cannot proceed

        # 3. Parse SASS and print FFMA info
        lines = sass_output.splitlines()
        print("\n--- FFMA Instructions ---")
        in_function = False
        func_name = "N/A"
        ffma_count = 0

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            if 'Function :' in stripped_line:
                func_name = stripped_line.lstrip('Function :').strip()
                in_function = True
                print(f"\nFunction: {func_name}")
                continue # Move to the next line

            if not in_function:
                continue # Skip lines before the first function header

            # Identify FFMA instruction lines (usually start with /* address */)
            if stripped_line.startswith("/*") and ' FFMA' in stripped_line:
                ffma_line = stripped_line
                ffma_count += 1

                # --- Extract Hex Values ---
                low_hex = 0
                high_hex = 0
                low_hex_comment = ""
                high_hex_comment = ""
                low_hex_error_msg = ""
                high_hex_error_msg = ""
                flags_found = False # Consolidated flag status

                # 1. Extract low_hex from the current FFMA line's comment
                try:
                    low_hex = extract_hex_from_line(ffma_line)
                    low_hex_comment = f"0x{low_hex:x}" # Format for consistency
                except ValueError as e:
                    low_hex_error_msg = f"(Error parsing low hex: {e})"

                # 2. Extract high_hex from the *next* line's comment
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    # Match a line that *only* contains a hex comment
                    match_high_hex = re.fullmatch(r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/', next_line)
                    if match_high_hex:
                        try:
                            high_hex_comment = match_high_hex.group(1)
                            high_hex = int(high_hex_comment, 16)
                        except ValueError:
                            high_hex_error_msg = f"(Error parsing high hex: {next_line})"
                    else:
                         high_hex_error_msg = f"(High hex line missing/unrecognized)" # next_line: '{next_line}'
                else:
                    high_hex_error_msg = "(High hex N/A - End of SASS)"

                # --- Determine Flags ---
                reuse_flag = False
                yield_flag = False
                reuse_flag_source = "N/A"
                yield_flag_source = "N/A"
                if not high_hex_error_msg:
                    # Reuse flag: Bit 58 in high_hex
                    reuse_flag = (high_hex & 0x0400000000000000) != 0
                    reuse_flag_source = f"HighHex[58]"

                    # Yield flag is bit 45 in high_hex (seems consistent)
                    yield_flag = (high_hex & 0x0000200000000000) != 0
                    yield_flag_source = f"HighHex[45]"

                # --- Print Results ---
                flag_details = []
                if not high_hex_error_msg:
                    flag_details.append(f"Reuse: {'Y' if reuse_flag else 'N'} [{reuse_flag_source}]")
                    flag_details.append(f"Yield: {'Y' if yield_flag else 'N'} [{yield_flag_source}]")
                    flag_details.append(f"HighHex: {high_hex_comment}")
                else:
                    flag_details.append(f"Reuse: ? ({high_hex_error_msg})") # Reuse depends on high hex now
                    flag_details.append(f"Yield: ? ({high_hex_error_msg})")

                print(f"  {ffma_line} ({', '.join(flag_details)})")


        if ffma_count == 0:
             print("No FFMA instructions found in the SASS output.")
        print("--- End FFMA Instructions ---")


# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile a CUDA file, extract SASS, and print FFMA instructions with Reuse/Yield flags.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--cu',
        required=True,
        help='Path to the input CUDA source file (.cu)'
    )
    parser.add_argument(
        '--arch',
        required=True,
        help='Target CUDA GPU architecture (e.g., sm_75, sm_86). Must be supported by your nvcc.'
    )

    args = parser.parse_args()

    # Validate input file existence
    if not os.path.isfile(args.cu):
         print(f"Error: Input file not found: {args.cu}")
         exit(1)

    # Validate arch format
    if not re.match(r'^sm_\d+$', args.arch):
        parser.error("Architecture must be in the format sm_XX (e.g., sm_86)")

    try:
        print_ffma_flags(args.cu, args.arch)
    except (EnvironmentError, FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback during debugging
        exit(1) 