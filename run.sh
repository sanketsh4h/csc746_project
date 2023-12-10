#!/bin/bash

# Compile the CUDA program
nvcc -o smith_waterman_cuda main.cu

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
    echo "Running with block dimensions: $dim"
    ./smith_waterman_cuda <<<"$dim"
    echo "--------------------------------------"

    # Run your CUDA program with ncu to collect metrics
    ncu --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg "$CUDA_PROGRAM" "$dim" >> "$METRICS_OUTPUT"
done

# Cleanup
rm smith_waterman_cuda
