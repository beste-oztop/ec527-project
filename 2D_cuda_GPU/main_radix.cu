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
 *    nvcc -o output_filename main_radix.cu 

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
 #define PI 3.14159265358979323846
 
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
 
 __device__ void bitReversal(complexFloat *data, int n) {
     unsigned int j = 0;
     for (unsigned int i = 0; i < n; i++) {
         if (i < j) {
             complexFloat temp = data[i];
             data[i] = data[j];
             data[j] = temp;
         }
         unsigned int mask = n >> 1;
         while (j & mask) {
             j &= ~mask;
             mask >>= 1;
         }
         j |= mask;
     }
 }
 
 __device__ complexFloat complexMul(complexFloat a, complexFloat b) {
     complexFloat result;
     result.re = a.re * b.re - a.im * b.im;
     result.im = a.re * b.im + a.im * b.re;
     return result;
 }
 
 __device__ complexFloat complexAdd(complexFloat a, complexFloat b) {
     complexFloat result;
     result.re = a.re + b.re;
     result.im = a.im + b.im;
     return result;
 }
 
 __device__ complexFloat complexSub(complexFloat a, complexFloat b) {
     complexFloat result;
     result.re = a.re - b.re;
     result.im = a.im - b.im;
     return result;
 }
 
 // Kernel for radix-2 FFT with bit reversal included
 __global__ void radix2FFTKernel(complexFloat *data, int n, int direction) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     if (tid >= n) return;
     
     __shared__ complexFloat shared_data[1024]; // Make sure this is large enough
     
     // Each thread loads one element to shared memory
     if (tid < n) {
         shared_data[threadIdx.x] = data[tid];
     }
     __syncthreads();
     
     // Perform bit reversal (only one thread per block should do this)
     if (threadIdx.x == 0) {
         bitReversal(shared_data, blockDim.x);
     }
     __syncthreads();
     
     // Butterfly computation
     for (int s = 1; s < blockDim.x; s *= 2) {
         int position = threadIdx.x;
         int butterfly_size = 2 * s;
         
         if ((position % butterfly_size) < s) {
             int partner = position + s;
             if (partner < blockDim.x) {
                 float angle = -direction * 2.0f * PI * (position % s) / butterfly_size;
                 complexFloat twiddle = {cosf(angle), sinf(angle)};
                 
                 complexFloat temp = complexMul(shared_data[partner], twiddle);
                 shared_data[partner] = complexSub(shared_data[position], temp);
                 shared_data[position] = complexAdd(shared_data[position], temp);
             }
         }
         __syncthreads();
     }
     
     // Write back
     if (tid < n) {
         data[tid] = shared_data[threadIdx.x];
     }
 }
 
 // Kernel for transposing a matrix
 __global__ void transposeKernel(complexFloat *input, complexFloat *output, int width, int height) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     if (x < width && y < height) {
         output[x * height + y] = input[y * width + x];
     }
 }
 
 // Kernel for row-wise 1D FFT
 __global__ void rowFFTKernel(complexFloat *data, int width, int height, int direction) {
     int row = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (row < height) {
         // Each thread processes one row
         complexFloat *row_data = &data[row * width];
         
         // Perform bit reversal
         bitReversal(row_data, width);
         
         // Butterfly computation
         for (int s = 1; s < width; s *= 2) {
             for (int i = 0; i < width; i += 2 * s) {
                 for (int j = 0; j < s; j++) {
                     if (i + j + s < width) {
                         float angle = -direction * 2.0f * PI * j / (2.0f * s);
                         complexFloat twiddle = {cosf(angle), sinf(angle)};
                         
                         complexFloat temp = complexMul(row_data[i + j + s], twiddle);
                         row_data[i + j + s] = complexSub(row_data[i + j], temp);
                         row_data[i + j] = complexAdd(row_data[i + j], temp);
                     }
                 }
             }
         }
     }
 }
 
 // Baseline DFT kernel
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
 
 int main(int argc, char *argv[]) {
     if (argc < 2) {
         fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
         return 1;
     }
 
     // GPU Timing variables
     cudaEvent_t start, stop;
     float elapsedTime;
 
     // Create CUDA events
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
 
     // Matrices for CUDA FFT
     complexFloat *d_data, *d_temp, *d_output;
     complexFloat *h_radix_result, *h_baseline_result;
 
     // Read the matrix input data
     complexFloat *data;
     int height, width;
     if (!readMatrix(argv[1], &data, &height, &width)) {
         return 1;
     }
 
     // Ensure the input dimensions are powers of 2
     if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0) {
         fprintf(stderr, "Error: Matrix dimensions must be powers of 2.\n");
         free(data);
         return 1;
     }
 
     // Allocate memory on the GPU
     CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(complexFloat) * height * width));
     CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp, sizeof(complexFloat) * height * width));
     CUDA_SAFE_CALL(cudaMalloc((void**)&d_output, sizeof(complexFloat) * height * width));
 
     // Copy input data to GPU
     CUDA_SAFE_CALL(cudaMemcpy(d_data, data, sizeof(complexFloat) * height * width, cudaMemcpyHostToDevice));
     
     // Allocate memory for results on host
     h_radix_result = (complexFloat *)malloc(sizeof(complexFloat) * height * width);
     h_baseline_result = (complexFloat *)malloc(sizeof(complexFloat) * height * width);
 
     // Begin timing for Radix-2 FFT
     cudaEventRecord(start, 0);
     
     // First perform row-wise FFT
     dim3 rowGridDim((height + 255) / 256, 1, 1);
     dim3 rowBlockDim(256, 1, 1);
     rowFFTKernel<<<rowGridDim, rowBlockDim>>>(d_data, width, height, 1);
     CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
     // Transpose the matrix
     dim3 transposeBlockDim(16, 16, 1);
     dim3 transposeGridDim((width + 15) / 16, (height + 15) / 16, 1);
     transposeKernel<<<transposeGridDim, transposeBlockDim>>>(d_data, d_temp, width, height);
     CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
     // Perform FFT on the transposed matrix (column-wise FFT)
     dim3 colGridDim((width + 255) / 256, 1, 1);
     dim3 colBlockDim(256, 1, 1);
     rowFFTKernel<<<colGridDim, colBlockDim>>>(d_temp, height, width, 1);
     CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
     // Transpose back to the original orientation
     transposeKernel<<<transposeGridDim, transposeBlockDim>>>(d_temp, d_data, height, width);
     CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsedTime, start, stop);
     printf("Radix-2 FFT time: %f ms\n", elapsedTime);
 
     // Copy Radix-2 FFT results back to host
     CUDA_SAFE_CALL(cudaMemcpy(h_radix_result, d_data, sizeof(complexFloat) * height * width, cudaMemcpyDeviceToHost));
 
     // Reset device data for baseline
     CUDA_SAFE_CALL(cudaMemcpy(d_data, data, sizeof(complexFloat) * height * width, cudaMemcpyHostToDevice));
     
     // Begin timing for baseline FFT
     cudaEventRecord(start, 0);
     
     // Row-wise baseline FFT
     dim3 baselineBlockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 baselineGridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     matrix1DFFTKernel<<<baselineGridDim, baselineBlockDim>>>(d_data, d_temp, nullptr, height, width, false);
     CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
     // Column-wise baseline FFT
     matrix1DFFTKernel<<<baselineGridDim, baselineBlockDim>>>(d_temp, d_output, nullptr, height, width, true);
     CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsedTime, start, stop);
     printf("Baseline FFT time: %f ms\n", elapsedTime);
 
     // Copy baseline results back to host
     CUDA_SAFE_CALL(cudaMemcpy(h_baseline_result, d_output, sizeof(complexFloat) * height * width, cudaMemcpyDeviceToHost));
 
     // Compare results
     if (compareResults(h_radix_result, h_baseline_result, height * width)) {
         printf("Results match!\n");
     } else {
         printf("Results do not match!\n");
         
         // Option to print the first few elements to debug
         printf("First 5 elements comparison:\n");
         for (int i = 0; i < 5; i++) {
             printf("Radix-2[%d]: %.6f + %.6fi\n", i, h_radix_result[i].re, h_radix_result[i].im);
             printf("Baseline[%d]: %.6f + %.6fi\n", i, h_baseline_result[i].re, h_baseline_result[i].im);
         }
     }
 
     // Write output to file
     FILE *outputFile = fopen("2DFFT_GPU_output.txt", "w");
     if (outputFile == NULL) {
         printf("Error: Could not open file for writing.\n");
         return 1;
     }
     for (int i = 0; i < height * width; i++) {
         fprintf(outputFile, "%f %f\n", h_radix_result[i].re, h_radix_result[i].im);
     }
     fclose(outputFile);
 
     // Free GPU memory
     CUDA_SAFE_CALL(cudaFree(d_data));
     CUDA_SAFE_CALL(cudaFree(d_temp));
     CUDA_SAFE_CALL(cudaFree(d_output));
 
     // Free host memory
     free(data);
     free(h_radix_result);
     free(h_baseline_result);
 
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
     bool match = true;
     int mismatch_count = 0;
     
     for (int i = 0; i < size; i++) {
         if (fabs(res1[i].re - res2[i].re) > epsilon || fabs(res1[i].im - res2[i].im) > epsilon) {
             match = false;
             mismatch_count++;
             if (mismatch_count <= 5) {
                 printf("Mismatch at index %d: (%.6f,%.6f) vs (%.6f,%.6f)\n", 
                        i, res1[i].re, res1[i].im, res2[i].re, res2[i].im);
             }
         }
     }
     
     if (!match) {
         printf("Total mismatches: %d out of %d elements\n", mismatch_count, size);
     }
     
     return match;
 }