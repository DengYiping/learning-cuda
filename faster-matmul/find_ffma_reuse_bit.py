#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import tempfile
import shutil

# --- Configuration (Copied from print_ffma_flags.py) ---
CUDA_HOME_ENV = os.getenv('CUDA_HOME')
if CUDA_HOME_ENV and os.path.exists(os.path.join(CUDA_HOME_ENV, 'bin', 'nvcc')):
    CUDA_HOME = CUDA_HOME_ENV
else:
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))
        print(f"Inferred CUDA_HOME='{CUDA_HOME}' from nvcc path.")
    else:
        CUDA_HOME = None

if not CUDA_HOME or not os.path.exists(os.path.join(CUDA_HOME, 'bin', 'nvcc')):
    raise EnvironmentError(
        "CUDA_HOME environment variable is not set or nvcc not found. "
        "Please set CUDA_HOME to your CUDA installation directory."
    )

NVCC_PATH = os.path.join(CUDA_HOME, 'bin', 'nvcc')
CUOBJDUMP_PATH = os.path.join(CUDA_HOME, 'bin', 'cuobjdump')

if not os.path.exists(NVCC_PATH):
     raise FileNotFoundError(f"nvcc not found at {NVCC_PATH}")
if not os.path.exists(CUOBJDUMP_PATH):
     raise FileNotFoundError(f"cuobjdump not found at {CUOBJDUMP_PATH}")

# --- Helper Functions (Copied from print_ffma_flags.py) ---

def run_command(command, description):
    """Runs a command and checks for errors."""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if result.stderr:
            print(f"STDERR ({description}):\n{result.stderr}")
        return result.stdout
    except FileNotFoundError as e:
        print(f"Error: Command not found - {e.filename}")
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
        # Return None instead of raising error, as not finding hex might be expected
        return None
    try:
        return int(match.group(1), 16)
    except ValueError:
        return None # Handle potential conversion errors

# --- Core Logic ---

def find_reuse_bit(cu_file, arch):
    """
    Compiles a .cu file, runs cuobjdump, parses SASS, and brute-forces
    to find the bit indicating the .reuse flag in FFMA instructions.
    """
    print(f"Analyzing {cu_file} for arch {arch} to find reuse bit.")
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
        ]
        try:
            run_command(compile_command, "NVCC Compile")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
             print(f"Compilation failed: {e}")
             return

        # 2. Run cuobjdump
        print(f"Running cuobjdump on {object_file}...")
        try:
            sass_output = run_cuobjdump(object_file)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"cuobjdump failed: {e}")
            return

        # 3. Parse SASS and collect FFMA data
        lines = sass_output.splitlines()
        ffma_data = []
        print("Parsing SASS for FFMA instructions...")

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Identify FFMA instruction lines
            if stripped_line.startswith("/*") and ' FFMA' in stripped_line:
                ffma_line = stripped_line
                low_hex = extract_hex_from_line(ffma_line)
                has_reuse = '.reuse' in ffma_line # Ground truth

                high_hex = None
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    # Check if next line is purely a hex comment
                    if re.fullmatch(r'/\*\s*0x[0-9a-fA-F]+\s*\*/', next_line):
                        high_hex = extract_hex_from_line(next_line)

                if low_hex is not None: # Only add if we have at least the low hex
                    ffma_data.append({
                        'sass_line': ffma_line,
                        'has_reuse': has_reuse,
                        'low_hex': low_hex,
                        'high_hex': high_hex # Can be None
                    })

        if not ffma_data:
            print("No FFMA instructions with valid hex found in SASS.")
            return

        print(f"Found {len(ffma_data)} FFMA instructions with low_hex.")

        # 4. Brute-force check bits 0-63
        print("Brute-forcing bits 0-63 for reuse flag correlation...")
        candidate_bits = []

        for bit_index in range(64):
            bit_mask = 1 << bit_index

            # Test low_hex
            low_hex_match = True
            for item in ffma_data:
                # Check if prediction matches ground truth
                predicted_reuse = (item['low_hex'] & bit_mask) != 0
                if predicted_reuse != item['has_reuse']:
                    low_hex_match = False
                    break
            if low_hex_match:
                candidate_bits.append({'bit': bit_index, 'source': 'low_hex'})

            # Test high_hex (only if all instructions have high_hex)
            high_hex_match = True
            all_have_high_hex = all(item['high_hex'] is not None for item in ffma_data)

            if all_have_high_hex:
                 for item in ffma_data:
                     # Check if prediction matches ground truth
                     predicted_reuse = (item['high_hex'] & bit_mask) != 0
                     if predicted_reuse != item['has_reuse']:
                         high_hex_match = False
                         break
                 if high_hex_match:
                      candidate_bits.append({'bit': bit_index, 'source': 'high_hex'})
            # Optional: Handle cases where *some* but not all have high_hex?
            # For simplicity, we require all instructions to have high_hex
            # to consider it a reliable source candidate.

        # 5. Output results
        print("--- Candidate Bits for Reuse Flag ---")
        if candidate_bits:
            for candidate in candidate_bits:
                print(f"  Bit: {candidate['bit']}, Source: {candidate['source']}")
        else:
            print("  No single bit in low_hex or high_hex consistently matched the .reuse flag.")
        print("--- End ---")


# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile CUDA, extract SASS, and brute-force bits to find the FFMA reuse flag.',
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

    if not os.path.isfile(args.cu):
         print(f"Error: Input file not found: {args.cu}")
         exit(1)

    if not re.match(r'^sm_\d+$', args.arch):
        parser.error("Architecture must be in the format sm_XX (e.g., sm_86)")

    try:
        find_reuse_bit(args.cu, args.arch)
    except (EnvironmentError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        exit(1) 