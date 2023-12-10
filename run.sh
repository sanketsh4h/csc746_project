#!/bin/bash

# Compile the CUDA program
nvcc -o smith_waterman_cuda main.cu

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
done

# Cleanup
rm smith_waterman_cuda
