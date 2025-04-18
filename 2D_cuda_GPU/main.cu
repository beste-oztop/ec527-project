/**
 *  This code implements a 2D Fast Fourier Transform (FFT) using CUDA.
 *  Request GPU node with this command:
 *  qrsh -l gpus=1 -P ec527
 *  do not forget to compile with the following command:
    module load cuda/11.3
    P100: -arch compute_60 -code sm_60
    V100: -arch compute_70 -code sm_70
    K40m: -arch compute_35 -code sm_35
 *
 *  Compile the code using the `nvcc` compiler:
 *    nvcc -o output_filename main.cu 

 *  Run the code with the input file as an argument:
 *    ./output_filename input_file_name
 * 
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8

typedef struct _complexFloat {
    float re;
    float im;
} complexFloat;

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int readMatrix(const char *filename, complexFloat **data, int *height, int *width);
double interval(struct timespec start, struct timespec end);
bool compareResults(complexFloat *res1, complexFloat *res2, int size);

__global__ void matrix1DFFTKernel(complexFloat *input, complexFloat *output, complexFloat *temp, int row, int col, bool row_or_col) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < row && j < col) {
        if (row_or_col == 0) { // rows
            complexFloat sum = {0, 0};
            float theta = 2 * M_PI / col * j;
            for (int jj = 0; jj < col; jj++) {
                complexFloat twiddle = {cos(theta * jj), sin(-theta * jj)};
                sum.re += input[i * col + jj].re * twiddle.re - input[i * col + jj].im * twiddle.im;
                sum.im += input[i * col + jj].im * twiddle.re + input[i * col + jj].re * twiddle.im;
            }
            temp[i * col + j] = sum;
        } else { // columns
            complexFloat sum = {0, 0};
            float theta = 2 * M_PI / row * i;
            for (int ii = 0; ii < row; ii++) {
                complexFloat twiddle = {cos(theta * ii), sin(-theta * ii)};
                sum.re += temp[ii * col + j].re * twiddle.re - temp[ii * col + j].im * twiddle.im;
                sum.im += temp[ii * col + j].im * twiddle.re + temp[ii * col + j].re * twiddle.im;
            }
            output[i * col + j] = sum;
        }
    }

}

    
__global__ void matrix1DFFTWithTransposeKernel(complexFloat *input, complexFloat *output, complexFloat *temp, int row, int col, bool row_or_col) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < row && j < col) {
        if (row_or_col == 0) { // rows
            complexFloat sum = {0, 0};
            float theta = 2 * M_PI / col * j;
            for (int jj = 0; jj < col; jj++) {
                complexFloat twiddle = {cos(theta * jj), sin(-theta * jj)};
                sum.re += input[i * col + jj].re * twiddle.re - input[i * col + jj].im * twiddle.im;
                sum.im += input[i * col + jj].im * twiddle.re + input[i * col + jj].re * twiddle.im;
            }
            temp[i * col + j] = sum;
        } else { // columns with transposing to increase efficiency
            complexFloat sum = {0, 0};
            float theta = 2 * M_PI / row * i;
            for (int ii = 0; ii < row; ii++) {
                complexFloat twiddle = {cos(theta * ii), sin(-theta * ii)};
                sum.re += temp[ii * col + j].re * twiddle.re - temp[ii * col + j].im * twiddle.im;
                sum.im += temp[ii * col + j].im * twiddle.re + temp[ii * col + j].re * twiddle.im;
            }
            output[i * col + j] = sum;
        }
    }
}


__global__ void rowWiseFFTKernelShared(complexFloat *input, complexFloat *output, int row, int col) {
    extern __shared__ complexFloat sharedData[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < row) {
        complexFloat sum = {0, 0};
        float theta = 2 * M_PI / col * j;
        
        // Process the input in tiles
        for (int tile = 0; tile < (col + blockDim.x - 1) / blockDim.x; tile++) {
            // Load this tile into shared memory
            int jTile = tile * blockDim.x + threadIdx.x;
            if (jTile < col) {
                sharedData[threadIdx.y * blockDim.x + threadIdx.x] = input[i * col + jTile];
            } else {
                sharedData[threadIdx.y * blockDim.x + threadIdx.x] = {0, 0};
            }
            __syncthreads();
            
            // Compute partial sum for this tile
            if (j < col) {
                for (int jj = 0; jj < blockDim.x && (tile * blockDim.x + jj) < col; jj++) {
                    int jPos = tile * blockDim.x + jj;
                    complexFloat twiddle = {cos(theta * jPos), sin(-theta * jPos)};
                    complexFloat element = sharedData[threadIdx.y * blockDim.x + jj];
                    sum.re += element.re * twiddle.re - element.im * twiddle.im;
                    sum.im += element.im * twiddle.re + element.re * twiddle.im;
                }
            }
            __syncthreads();
        }
        
        if (j < col) {
            output[i * col + j] = sum;
        }
    }
}

__global__ void columnWiseFFTKernelShared(complexFloat *input, complexFloat *output, int row, int col) {
    extern __shared__ complexFloat sharedData[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j < col) {
        complexFloat sum = {0, 0};
        float theta = 2 * M_PI / row * i;
        
        // Process the input in tiles
        for (int tile = 0; tile < (row + blockDim.y - 1) / blockDim.y; tile++) {
            // Load this tile into shared memory
            int iTile = tile * blockDim.y + threadIdx.y;
            if (iTile < row) {
                sharedData[threadIdx.y * blockDim.x + threadIdx.x] = input[iTile * col + j];
            } else {
                sharedData[threadIdx.y * blockDim.x + threadIdx.x] = {0, 0};
            }
            __syncthreads();
            
            // Compute partial sum for this tile
            if (i < row) {
                for (int ii = 0; ii < blockDim.y && (tile * blockDim.y + ii) < row; ii++) {
                    int iPos = tile * blockDim.y + ii;
                    complexFloat twiddle = {cos(theta * iPos), sin(-theta * iPos)};
                    complexFloat element = sharedData[ii * blockDim.x + threadIdx.x];
                    sum.re += element.re * twiddle.re - element.im * twiddle.im;
                    sum.im += element.im * twiddle.re + element.re * twiddle.im;
                }
            }
            __syncthreads();
        }
        
        if (i < row) {
            output[i * col + j] = sum;
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // GPU Timing variables
    cudaEvent_t start, stop;
    float elapsedTime;

    // Matrices for CUDA FFT
    complexFloat *d_input, *d_output, *d_temp;

    // Read the matrix input data
    complexFloat *data, *result1, *result2, *result3;
    int height, width;
    if (!readMatrix(argv[1], &data, &height, &width)) {
        return 1;
    }

    // Allocate memory for results
    result1 = (complexFloat *)malloc(sizeof(complexFloat) * height * width);
    result2 = (complexFloat *)malloc(sizeof(complexFloat) * height * width);
    result3 = (complexFloat *)malloc(sizeof(complexFloat) * height * width);

    // Allocate memory on the GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_input, sizeof(complexFloat) * height * width));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_output, sizeof(complexFloat) * height * width));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp, sizeof(complexFloat) * height * width));

    // Copy input data to GPU
    CUDA_SAFE_CALL(cudaMemcpy(d_input, data, sizeof(complexFloat) * height * width, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // 1. Global memory access kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrix1DFFTKernel<<<gridDim, blockDim>>>(d_input, d_output, d_temp, height, width, 0); // Row-wise FFT
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    matrix1DFFTKernel<<<gridDim, blockDim>>>(d_input, d_output, d_temp, height, width, 1); // Column-wise FFT
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Global memory access kernel time: %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaMemcpy(result1, d_output, sizeof(complexFloat) * height * width, cudaMemcpyDeviceToHost));

    // 2. Transposing the columns kernel
    cudaEventRecord(start, 0);

    matrix1DFFTWithTransposeKernel<<<gridDim, blockDim>>>(d_input, d_output, d_temp, height, width, 0); // Row-wise FFT
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    matrix1DFFTWithTransposeKernel<<<gridDim, blockDim>>>(d_input, d_output, d_temp, height, width, 1); // Column-wise FFT with transpose
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Transposing columns kernel time: %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaMemcpy(result2, d_output, sizeof(complexFloat) * height * width, cudaMemcpyDeviceToHost));

    // 3. Row-column kernels pair
    cudaEventRecord(start, 0);

    size_t sharedMemSize = blockDim.x * blockDim.y * sizeof(complexFloat);
    rowWiseFFTKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_temp, height, width); // Row-wise FFT
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    columnWiseFFTKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_temp, d_output, height, width); // Column-wise FFT
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Row-column kernels pair time: %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaMemcpy(result3, d_output, sizeof(complexFloat) * height * width, cudaMemcpyDeviceToHost));
    printf("\n**Results**\n");
    printf("Global vs Transposing: %s\n", compareResults(result1, result2, height * width) ? "Match" : "Mismatch");
    printf("Global vs Row-Column: %s\n", compareResults(result1, result3, height * width) ? "Match" : "Mismatch");
    printf("Transposing vs Row-Column: %s\n", compareResults(result2, result3, height * width) ? "Match" : "Mismatch");

    // Write output to file
    FILE *outputFile = fopen("2DFFT_GPU_output.txt", "w");
    if (outputFile == NULL) {
        printf("Error: Could not open file for writing.\n");
        return 1;
    }
    for (int i = 0; i < height * width; i++) {
        fprintf(outputFile, "%f %f\n", result1[i].re, result1[i].im);
    }
    fclose(outputFile);

    // Free GPU memory
    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_output));
    CUDA_SAFE_CALL(cudaFree(d_temp));

    // Free host memory
    free(data);
    free(result1);
    free(result2);
    free(result3);

    return 0;
}


double interval(struct timespec start, struct timespec end) {
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec -= 1;
        temp.tv_nsec += 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}

int readMatrix(const char *filename, complexFloat **data, int *height, int *width) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return 0;
    }

    char line[1024];
    int h = 0, w = 0;
    char *token;

    // First pass: Determine dimensions
    while (fgets(line, sizeof(line), file) != NULL) {
        int count = 0;
        token = strtok(line, " \t\n");
        while (token) {
            count++;
            token = strtok(NULL, " \t\n");
        }
        if (count > 0) {
            if (h == 0) {
                w = count;
            } else if (count != w) {
                fprintf(stderr, "Error: Inconsistent number of columns in matrix rows.\n");
                fclose(file);
                return 0;
            }
            h++;
        }
    }

    if (h == 0 || w == 0) {
        fprintf(stderr, "Error: No valid data found in file.\n");
        fclose(file);
        return 0;
    }

    *height = h;
    *width = w;
    *data = (complexFloat *)malloc(h * w * sizeof(complexFloat));
    if (!*data) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(file);
        return 0;
    }

    // Second pass: Read data
    rewind(file);
    int idx = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        token = strtok(line, " \t\n");
        while (token) {
            if (idx >= h * w) {
                fprintf(stderr, "Error: More data than expected.\n");
                free(*data);
                fclose(file);
                return 0;
            }
            (*data)[idx].re = atof(token);
            (*data)[idx].im = 0.0;
            idx++;
            token = strtok(NULL, " \t\n");
        }
    }

    if (idx != h * w) {
        fprintf(stderr, "Error: Less data than expected.\n");
        free(*data);
        fclose(file);
        return 0;
    }

    fclose(file);
    return 1;
}

bool compareResults(complexFloat *res1, complexFloat *res2, int size) {
    const float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {
        if (fabs(res1[i].re - res2[i].re) > epsilon || fabs(res1[i].im - res2[i].im) > epsilon) {
            return false;
        }
    }
    return true;
}