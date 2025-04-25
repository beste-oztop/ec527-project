#!/bin/bash

EXECUTABLE="./main_radix_block_size"
INPUT_FILE="fft_input.txt"
OUTPUT_DIR="block_size_results"
mkdir -p $OUTPUT_DIR

# Run the CUDA program with different block sizes

for BLOCK_SIZE in 8 16 32 64 128 256 512 1024; do
    echo "Running with block size: $BLOCK_SIZE"
    OUTPUT_FILE="$OUTPUT_DIR/output_block_${BLOCK_SIZE}.txt"
    
    nvcc -o main_radix_block_size -D BLOCK_SIZE=$BLOCK_SIZE main_radix.cu 

    $EXECUTABLE $INPUT_FILE > $OUTPUT_FILE
    
    echo "Results saved to $OUTPUT_FILE"
done

echo "All runs completed."