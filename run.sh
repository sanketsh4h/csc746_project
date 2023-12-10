#!/bin/bash

# Compile the CUDA program
nvcc main.cu -o smith_waterman_cuda

# Path to your CUDA program
CUDA_PROGRAM="./smith_waterman_cuda"

# Output file for metrics
METRICS_OUTPUT="metrics_output.txt"

# Clear the metrics output file
> "$METRICS_OUTPUT"

# Define the input sequences
sequence1="AGTACGTA"
sequence2="TATAGCGA"

# Define an array of block dimensions
block_dimensions=("8 8" "16 16" "32 32")

# Loop through each block dimension and run the program
for dim in "${block_dimensions[@]}"; do
    IFS=' ' read -r blockDimX blockDimY <<< "$dim"
    echo "Running with block dimensions: $dim"
    $CUDA_PROGRAM $blockDimX $blockDimY
    ncu --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all $CUDA_PROGRAM $blockDimX $blockDimY >> "$METRICS_OUTPUT"
    echo "--------------------------------------"
done


# Cleanup
rm smith_waterman_cuda
