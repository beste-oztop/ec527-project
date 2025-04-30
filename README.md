# ec527-project
This project was done by Beste Oztop, Zeynep Ece Kizilates, and Amin Khodaverdian.

Within each directory and in the main.c files you can find instructions on how to run and replicate the outputs.
The following is a detailed explanation of each directory:
- 1dfft_CPU: It contains the code for Radix-2, Radix-4, and Radix-8 serial CPU implementations.
- 1dfft_pthread: This directory contains several files, each of which defines: 
    - main.c: It contains the main Pthread implementation for all three methods.
    - main_cache.c: This code also applies the perf model on the functions to calculate the cache efficiency for each of our methods.
    - timing_measurements: This code contains the timing simulation for our threading.

**NOTE: In the timing_measurements code, we have an input generator that allows you to define the size of your input.**

- 2D_cuda_GPU: main.cu is our main GPU implementation for Radix-2, DFT baseline with and without shared memory.
- 2D_first_attempt: Our first attempt at writing a baseline to compare with outputs from MATLAB.
- 2D_PThread: We also tried using Pthread for 2D FFT, which, based on Prof. Herbordt's comment, led us to change our threading to 1D FFT.
- 2D_Threading: We used OpenMP for 2D FFT (we left it because we focused on 1D FFT based on Prof. Herbordt's feedback).
- Radix_2_4_8: Our first implementation of different methods (you can refer to 1dfft_CPU for a full implementation)


