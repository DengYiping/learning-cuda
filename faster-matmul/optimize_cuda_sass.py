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
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # print(f"STDOUT ({description}):\n{result.stdout}") # Optional: print stdout
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

# --- Core Logic (mostly unchanged from original script) ---

def run_cuobjdump(file_path):
    """Runs cuobjdump to get SASS."""
    command = [CUOBJDUMP_PATH, '-sass', file_path]
    # Use the general run_command function for consistency
    # Use check=False here because we handle the error in the caller if needed,
    # but the original script asserted returncode 0, so let's keep that check
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running cuobjdump:")
        print(f"Command: {' '.join(command)}")
        print(f"Return Code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    return result.stdout

def extract_ffma(sass):
    lines = sass.splitlines()
    collected = []
    current = []

    arch_name, func_name = 'N/A', 'N/A'
    skip_next_line = False
    for line in lines:
        if 'code for' in line:
            arch_name = line.lstrip().lstrip('code for ').rstrip().split(' ')[0] # Get only sm_XX part
        elif 'Function :' in line:
            func_name = line.lstrip().lstrip('Function :').rstrip()
            # Reset current segment when a new function starts
            if len(current) >= 16:
                 assert len(current) % 2 == 0
                 collected.append((f'{arch_name}::{func_name}_prev', current)) # Name previous segment
            current = []
        elif 'FFMA' in line:
            current.append(line)
            skip_next_line = True
        elif skip_next_line:
            current.append(line)
            skip_next_line = False
        else:
            if len(current) >= 16: # Minimum length for a potential segment
                assert len(current) % 2 == 0
                collected.append((f'{arch_name}::{func_name}', current))
            current = [] # Reset segment if non-FFMA line encountered

    # Add the last collected segment if any
    if len(current) >= 16:
        assert len(current) % 2 == 0
        collected.append((f'{arch_name}::{func_name}', current))

    if os.getenv('DG_PRINT_REG_REUSE', None):
        print(f'Found {len(collected)} FFMA segments')
    return collected


def extract_hex_from_line(line):
    match = re.search(r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/', line)
    if not match:
        raise ValueError(f"Could not find hex pattern /* 0x... */ in line: {line}")
    return int(match.group(1), 16)


def validate(m, offset, le_bytes, num_lines):
    segment_len = num_lines // 2
    if len(le_bytes) != segment_len:
         print(f"Warning: validate length mismatch. Expected {segment_len} byte sequences, got {len(le_bytes)}.")
         return False # Should not happen if called correctly
    # Ensure we don't read past the end of the mapped file
    if offset + segment_len * 16 > len(m):
        # print(f"DEBUG: Validation failed - offset {offset} + segment_len {segment_len}*16 exceeds mmap length {len(m)}")
        return False

    # Check first instruction pair
    if m[offset:offset + 16] != le_bytes[0]:
         # print(f"DEBUG: First byte mismatch at offset {offset}")
         # print(f"DEBUG: Expected: {le_bytes[0].hex()}")
         # print(f"DEBUG: Found:    {m[offset:offset+16].hex()}")
         return False

    # Check subsequent instruction pairs
    for i in range(1, segment_len):
        start = offset + i * 16
        end = start + 16
        if m[start:end] != le_bytes[i]:
            # print(f"DEBUG: Mismatch at index {i}, offset {start}")
            # print(f"DEBUG: Expected: {le_bytes[i].hex()}")
            # print(f"DEBUG: Found:    {m[start:end].hex()}")
            return False
    # print(f"DEBUG: Validation successful at offset {offset} for {segment_len} pairs.")
    return True


def parse_registers(line):
    # Remove comments and semicolons
    line = re.sub(r'/\*.*?\*/', '', line)
    line = line.replace(';', '').strip()
    # Split operands
    parts = line.split() # e.g., ['FFMA', 'R6,', 'R8,', 'R9,', 'R7']
    registers = []
    # Iterate through parts, looking for registers (start with 'R' followed by digits)
    for part in parts:
        token = part.strip(',') # Remove trailing comma if present
        if re.match(r'^R\d+$', token):
            registers.append(token)
        elif re.match(r'^R\d+\.reuse$', token): # Handle .reuse suffix if present
            registers.append(token.split('.')[0])

    # Example FFMA format: FFMA RDest, RSrc1, RSrc2, RAdd;
    # We expect at least 4 registers. The destination is usually the first one listed.
    # However, the original code checked the *second to last* register parsed. Let's stick to that logic.
    # It might be specific to how cuobjdump formats FFMA or the specific patterns it targets.
    # Let's re-examine the original `parse_registers` carefully.
    # Original logic: split by comma, then by space, then check if word starts with 'R'.
    # This seems fragile. Let's refine based on typical SASS output.
    # FFMA R6, R8, R9, R7 ; /* 0x... */ -> R6, R8, R9, R7
    # FFMA.reuse R6, R8, R9, R7 ; /* 0x... */ -> R6, R8, R9, R7

    # Let's try a regex approach for robustness:
    # Match things like R<digits> potentially followed by .<something>
    reg_matches = re.findall(r'(R\d+)(\.\w+)?', line)
    registers = [match[0] for match in reg_matches] # Get only the R<digits> part

    if len(registers) < 4:
         print(f"Warning: Could not parse expected registers from line: {line}")
         print(f"Found registers: {registers}")
         # Return placeholders or raise error? Let's return placeholders for now.
         return ["R?", "R?", "R?", "R?"]

    # The original code used `registers[-2]`. Let's keep that specific index.
    # For `FFMA R6, R8, R9, R7;` this would be `R9`. This seems unusual for a destination register.
    # Let's assume the original author had a reason or it targeted a specific pattern.
    # **Correction:** Looking again, `tokens = line.strip().split(',')` then iterates `words` in `token.split()`.
    # Example: `FFMA R6, R8, R9, R7 ;` -> `tokens = ['FFMA R6', ' R8', ' R9', ' R7 ']`
    # token='FFMA R6' -> words=['FFMA', 'R6'] -> registers=['R6']
    # token=' R8' -> words=['R8'] -> registers=['R6', 'R8']
    # token=' R9' -> words=['R9'] -> registers=['R6', 'R8', 'R9']
    # token=' R7 ' -> words=['R7'] -> registers=['R6', 'R8', 'R9', 'R7']
    # So `registers[-2]` IS indeed `R9` in this common case.

    # Let's reimplement the original parsing logic more clearly:
    line = re.sub(r'/\*.*?\*/', '', line)
    line = line.replace(';', '').strip()
    tokens = line.split(',')
    parsed_registers = []
    for token in tokens:
        words = token.strip().split()
        for word in words:
            if word.startswith('R') and word.split('.')[0][1:].isdigit(): # Check R followed by digits
                reg = word.split('.')[0] # Remove .reuse etc.
                parsed_registers.append(reg)

    if not parsed_registers:
         raise ValueError(f"No registers found in line: {line}")

    # print(f"DEBUG: Parsed registers for line '{line}': {parsed_registers}")
    return parsed_registers


def modify_segment(m, name, ffma_lines):
    # Only process segments with at least 16 FFMA lines (8 instruction pairs)
    # as per the original logic's check `if len(current) >= 16:` before adding to `collected`
    # And the original code truncated based on `num_lines = (len(ffma_lines) * 9 // 16) // 2 * 2`
    # Let's simplify: process all valid pairs found. The `validate` function ensures we match the whole block.
    if len(ffma_lines) < 2 or len(ffma_lines) % 2 != 0:
        print(f" > segment `{name}` skipped: Invalid number of lines ({len(ffma_lines)})")
        return 0 # No modifications made

    num_pairs = len(ffma_lines) // 2
    num_lines = num_pairs * 2 # Ensure we only process pairs

    le_bytes, new_le_bytes = [], []
    reused_indices_after_modification = []
    dst_reg_set = set()
    last_reused, last_dst_reg = False, ''
    num_changed = 0

    # Extract original bytes and determine modifications
    for i in range(num_pairs):
        low_line, high_line = ffma_lines[i * 2], ffma_lines[i * 2 + 1]
        try:
            # Get registers. Original script uses [-2] which seems to be the 3rd source operand (e.g., R9 in FFMA R6, R8, R9, R7)
            # Let's stick to it, assuming it was intentional.
            # We need the *destination* register though to track reuse chains. Let's assume it's the first one parsed.
            parsed_regs = parse_registers(low_line)
            if not parsed_regs:
                 raise ValueError("Could not parse registers for reuse logic.")
            dst_reg = parsed_regs[0] # Assume first register is destination for reuse tracking

            low_hex = extract_hex_from_line(low_line)
            high_hex = extract_hex_from_line(high_line)
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping modification for pair {i} in segment '{name}' due to parsing error: {e}")
            print(f"  Line1: {low_line}")
            print(f"  Line2: {high_line}")
            # Cannot proceed with this segment if parsing fails, as byte sequences will be wrong
            return 0 # Indicate no modifications done for this segment attempt

        original_bytes = low_hex.to_bytes(8, 'little') + high_hex.to_bytes(8, 'little')
        le_bytes.append(original_bytes)

        # Determine modification
        modified_high_hex = high_hex
        reused_flag_is_set = (high_hex & 0x0400000000000000) != 0
        current_pair_reused = False # Track if this pair *remains* reused after potential modification

        if reused_flag_is_set:
            is_first_occurrence = dst_reg not in dst_reg_set
            is_continuation = (last_reused and dst_reg == last_dst_reg)

            # Original condition to *clear* the reuse bits:
            # if is_first_occurred or (last_reused and dst_reg == last_dst_reg):
            if is_first_occurrence or is_continuation:
                # Check if the expected bits are actually set before clearing
                if high_hex & 0x0400200000000000: # reuse bit + yield bit
                    modified_high_hex ^= 0x0400200000000000 # Clear reuse and yield bits
                    num_changed += 1
                    current_pair_reused = False # It's no longer marked as reused
                    # print(f"DEBUG: Modifying pair {i} for dst_reg {dst_reg}. FirstOcc={is_first_occurrence}, Cont={is_continuation}")
                else:
                    # This case was asserted in the original code. Let's warn instead.
                     print(f"Warning: Expected reuse+yield bits (0x08002000...) not set in high_hex {hex(high_hex)} for pair {i}, dst_reg {dst_reg} in segment '{name}'. No change made.")
                     current_pair_reused = True # Remains reused because we didn't change it
            else:
                 # This instruction *is* reused and it's *not* the first time or a continuation
                 # The original code added index `i` to `reused_list` here.
                 reused_indices_after_modification.append(i)
                 current_pair_reused = True
        else:
             # Reuse flag was not set initially
             current_pair_reused = False

        # Add destination register to set *after* checking first occurrence
        dst_reg_set.add(dst_reg)
        # Record state for the next iteration
        last_reused, last_dst_reg = current_pair_reused, dst_reg

        # Store the potentially modified bytes
        new_le_bytes.append(low_hex.to_bytes(8, 'little') + modified_high_hex.to_bytes(8, 'little'))

    if num_changed == 0:
        if os.getenv('DG_PRINT_REG_REUSE', None):
            print(f" > segment `{name}`: No modifications needed.")
        return 0

    if os.getenv('DG_PRINT_REG_REUSE', None):
        print(f' > segment `{name}` requires modification ({num_changed} changed pairs). New reused list indices: {reused_indices_after_modification}')

    # Find the sequence of original bytes in the memory map
    total_modified_in_file = 0
    search_pos = 0
    while True:
        offset = m.find(le_bytes[0], search_pos)
        if offset == -1:
            break # Not found anymore

        # Validate the entire sequence at this offset
        if validate(m, offset, le_bytes, num_lines):
            if os.getenv('DG_PRINT_REG_REUSE', None):
                 print(f"   - Found valid sequence at offset {hex(offset)}. Applying modification.")
            # Replace with the modified bytes
            try:
                for i in range(num_pairs):
                    start = offset + i * 16
                    end = start + 16
                    m[start:end] = new_le_bytes[i]
                total_modified_in_file += 1
                search_pos = offset + 16 # Start next search after the first instruction pair of the *current* match
            except Exception as e:
                print(f"Error modifying mmap at offset {hex(offset + i * 16)}: {e}")
                # Continue searching, maybe this occurrence was partially overlapping another?
                search_pos = offset + 1
        else:
            # If validation fails, move search position past this potential start
            search_pos = offset + 1

    if total_modified_in_file == 0:
         print(f"Warning: Segment '{name}' was marked for modification, but the byte sequence was not found/validated in the file.")
    elif os.getenv('DG_PRINT_REG_REUSE', None):
         print(f"   - Modified {total_modified_in_file} occurrences in the file for segment '{name}'.")

    return total_modified_in_file


def process(cu_file, output_executable, arch, keep_obj=False):
    """Compiles, modifies SASS, and links a CUDA source file."""
    print(f"Processing {cu_file} for arch {arch} -> {output_executable}")

    base_name = os.path.splitext(os.path.basename(cu_file))[0]
    # Use a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        object_file = os.path.join(tmpdir, f"{base_name}.o")

        # 1. Compile .cu to .o
        #    -gencode specifies the virtual arch (compute_XX) and real arch (sm_XX)
        #    Use the same arch for both for simplicity unless PTX is needed separately
        #    Adding -O3 for optimization, -g for debug info (optional but good practice)
        compile_command = [
            NVCC_PATH,
            '-gencode', f'arch=compute_{arch.split("_")[1]},code=sm_{arch.split("_")[1]}', # e.g., arch=compute_86,code=sm_86
            '-c', cu_file,
            '-o', object_file,
            '-O3', # Add optimization
            # '-g', # Optional: Add debug symbols
            # '-lineinfo' # Optional: Add line info
        ]
        run_command(compile_command, "NVCC Compile")

        # 2. Run cuobjdump on the object file
        print(f"Running cuobjdump on {object_file}")
        try:
             sass_output = run_cuobjdump(object_file)
        except subprocess.CalledProcessError:
             print("cuobjdump failed. Cannot extract SASS. Linking unmodified object file.")
             # Link the original object file if SASS dump fails
             link_command = [
                 NVCC_PATH,
                 '-gencode', f'arch=compute_{arch.split("_")[1]},code=sm_{arch.split("_")[1]}',
                 object_file,
                 '-o', output_executable,
                 '-lcudart' # Link against CUDA runtime
             ]
             run_command(link_command, "NVCC Link (Unmodified)")
             print(f"Warning: Output executable '{output_executable}' created without SASS modifications.")
             # Copy object file out if requested
             if keep_obj:
                 final_obj_path = f"{base_name}_unmodified.o"
                 shutil.copy2(object_file, final_obj_path)
                 print(f"Unmodified object file saved as: {final_obj_path}")
             return # Stop processing here

        # 3. Extract FFMA segments from SASS
        segments = extract_ffma(sass_output)
        if not segments:
             print("No relevant FFMA segments found in SASS. Linking unmodified object file.")
             # Link the original object file if no segments found
             link_command = [
                 NVCC_PATH,
                 '-gencode', f'arch=compute_{arch.split("_")[1]},code=sm_{arch.split("_")[1]}',
                 object_file,
                 '-o', output_executable,
                 '-lcudart' # Link against CUDA runtime
             ]
             run_command(link_command, "NVCC Link (Unmodified)")
             print(f"Output executable '{output_executable}' created without SASS modifications.")
              # Copy object file out if requested
             if keep_obj:
                 final_obj_path = f"{base_name}_unmodified.o"
                 shutil.copy2(object_file, final_obj_path)
                 print(f"Unmodified object file saved as: {final_obj_path}")
             return # Stop processing here


        # 4. Modify the object file in place using mmap
        print(f"Attempting to modify SASS in {object_file}")
        total_modifications = 0
        try:
            with open(object_file, 'r+b') as f:
                # Check file size before mmap
                file_size = os.fstat(f.fileno()).st_size
                if file_size == 0:
                    print(f"Error: Object file '{object_file}' is empty. Cannot modify.")
                    raise ValueError("Object file is empty")

                # Memory map the file
                # Use ACCESS_WRITE for modification
                # On Windows, length must be specified for ACCESS_WRITE, 0 means map entire file
                # On Linux/macOS, 0 works for both read and write access
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)

                try:
                    for name, lines in segments:
                        total_modifications += modify_segment(mm, name, lines)
                    if total_modifications > 0:
                         print(f"Total modifications applied to {object_file}: {total_modifications}")
                    else:
                         print(f"No modifications were ultimately applied to {object_file}.")
                finally:
                    # Ensure mmap is closed and flushed
                    mm.flush()
                    mm.close()

        except Exception as e:
            print(f"Error during SASS modification of {object_file}: {e}")
            print("Linking unmodified object file as a fallback.")
            # Link the original (or potentially partially modified but unsafe) object file
            link_command = [
                NVCC_PATH,
                '-gencode', f'arch=compute_{arch.split("_")[1]},code=sm_{arch.split("_")[1]}',
                object_file, # This might be the original or partially modified one
                '-o', output_executable,
                '-lcudart'
            ]
            run_command(link_command, "NVCC Link (Fallback)")
            print(f"Warning: Output executable '{output_executable}' may not contain intended SASS modifications.")
             # Copy object file out if requested
            if keep_obj:
                final_obj_path = f"{base_name}_failed_modification.o"
                shutil.copy2(object_file, final_obj_path)
                print(f"Object file (modification attempted) saved as: {final_obj_path}")
            return # Stop processing


        # Copy the modified object file out of temp dir if requested *before* linking
        if keep_obj:
             final_obj_path = f"{base_name}_modified.o"
             shutil.copy2(object_file, final_obj_path)
             print(f"Modified object file saved as: {final_obj_path}")

        # 5. Link the modified object file into an executable
        link_command = [
            NVCC_PATH,
            '-gencode', f'arch=compute_{arch.split("_")[1]},code=sm_{arch.split("_")[1]}',
            object_file, # This is the modified object file
            '-o', output_executable,
            '-lcudart' # Link against CUDA runtime
        ]
        run_command(link_command, "NVCC Link (Modified)")

        print(f"Successfully created executable '{output_executable}' with SASS modifications.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile CUDA code, apply FFMA register reuse optimization to SASS, and link.',
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
    parser.add_argument(
        '--out',
        required=True,
        help='Path for the output executable file.'
    )
    parser.add_argument(
        '--keep-obj',
        action='store_true',
        help='Keep the intermediate (modified or unmodified) object file.'
    )
    parser.add_argument(
        '--debug-print',
        action='store_true',
        help='Enable debug printing for register reuse logic (sets DG_PRINT_REG_REUSE=1).'
    )

    args = parser.parse_args()

    # Validate arch format
    if not re.match(r'^sm_\d+$', args.arch):
        parser.error("Architecture must be in the format sm_XX (e.g., sm_86)")

    # Set environment variable for debug printing if requested
    if args.debug_print:
        os.environ['DG_PRINT_REG_REUSE'] = '1'
    elif 'DG_PRINT_REG_REUSE' in os.environ:
         # Ensure it's unset if flag not provided but env var exists from environment
         del os.environ['DG_PRINT_REG_REUSE']


    try:
        process(args.cu, args.out, args.arch, args.keep_obj)
    except (EnvironmentError, FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        exit(1)