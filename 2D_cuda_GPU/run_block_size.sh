#!/bin/bash

EXECUTABLE="./main_radix_block_size"
OUTPUT_DIR="block_size_results"
mkdir -p $OUTPUT_DIR

# Run the CUDA program with different block sizes

for FFT_LENGTH in 1024 2048 4096 8192 16384; do
    echo "Running with FFT length: $FFT_LENGTH"
    for BLOCK_SIZE in 8 16 32 64 128 256 512 1024; do
        echo "Running with block size: $BLOCK_SIZE"
        OUTPUT_FILE="$OUTPUT_DIR/output_block_${BLOCK_SIZE}_length_${FFT_LENGTH}.txt"
        
        nvcc -o main_radix_block_size -D BLOCK_SIZE=$BLOCK_SIZE main_radix.cu 

        $EXECUTABLE $FFT_LENGTH > $OUTPUT_FILE
        
        echo "Results saved to $OUTPUT_FILE"
    done
done
echo "All runs completed."