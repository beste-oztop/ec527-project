#! /bin/bash

EXECUTABLE="./main_radix_epsilon"
OUTPUT_DIR="epsilon_results"
mkdir -p $OUTPUT_DIR

# Run the CUDA program with different epsilon values

for FFT_LENGTH in 1024 2048 4096 8192 16384; do
    echo "Running with FFT length: $FFT_LENGTH"
    for EPSILON in 0.01 0.025 0.05 0.1 0.25 0.5 0.75 1.0 1.25; do
        echo "Running with epsilon: $EPSILON"
        OUTPUT_FILE="$OUTPUT_DIR/output_epsilon_${EPSILON}_length_${FFT_LENGTH}.txt"
        
        nvcc -o main_radix_epsilon -D EPSILON=$EPSILON main_radix.cu 

        $EXECUTABLE $FFT_LENGTH > $OUTPUT_FILE
        
        echo "Results saved to $OUTPUT_FILE"
    done
done
echo "All runs completed."