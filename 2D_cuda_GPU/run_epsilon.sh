#! /bin/bash

EXECUTABLE="./main_radix_epsilon"
INPUT_FILE="fft_input.txt"
OUTPUT_DIR="epsilon_results"
mkdir -p $OUTPUT_DIR



# Run the CUDA program with different epsilon values
for EPSILON in 0.01 0.025 0.05 0.1 0.25 0.5 0.75 1.0 1.25; do
    echo "Running with epsilon: $EPSILON"
    OUTPUT_FILE="$OUTPUT_DIR/output_epsilon_${EPSILON}.txt"
    
    nvcc -o main_radix_epsilon -D EPSILON=$EPSILON main_radix.cu 

    $EXECUTABLE $INPUT_FILE > $OUTPUT_FILE
    
    echo "Results saved to $OUTPUT_FILE"
done
echo "All runs completed."