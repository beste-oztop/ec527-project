/* perf stat -e cache-references,cache-misses ./fft_main 2 12 1 
gcc -pthread timing_measurements.c -lpthread -lm -lrt -o fft_main
OMP_NUM_THREADS=8 ./fft_main fft_input.txt 2 12 1
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>

typedef unsigned int u32;
typedef unsigned short u16;
typedef char u8;
#define DIT 0 
#define DIF 1 
#define NUM_TESTS 28
#define OPTIONS 8
//#define NUM_THREADS 5

typedef struct _complexFloat{
    float re;
    float im;
}complexFloat;


struct thread_data {
    int thread_id;
    int num_threads;
    complexFloat *input;
    complexFloat *output;
    complexFloat *butterfly; 
    bool IFFT;
    int fftRadix;
    int fftStage;
    int fftLength;
    int fftType;
    pthread_barrier_t *barrier;
};

pthread_mutex_t mutexA;   /* declare a global mutex */

void* fft_radix2_DIF_thread(void *threadarg);
void* fft_radix2_DIT_thread(void *threadarg);
void* fft_radix4_DIT_thread(void *threadarg);
void* fft_radix4_DIF_thread(void *threadarg);
void* fft_radix8_DIT_thread(void *threadarg);
void* fft_radix8_DIF_thread(void *threadarg);
int fft_radix2_DIT(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix2_DIF(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix4_DIT(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix4_DIF(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix8_DIT(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix8_DIF(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int readMatrix(const char *filename, complexFloat **data, int *height, int *width);
static complexFloat complexAdd( complexFloat A, complexFloat B);
static complexFloat complexSub( complexFloat A, complexFloat B);
static complexFloat complexMul( complexFloat A, complexFloat B);
static u32 reverseBit( u32 n, int fftRadix, int fftLength);
static complexFloat getWeight( int iButterfly, int indexWeight, int fftLength);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}


int main(int argc, char *argv[]) {
    long i, OPTION;
    struct timespec time_start, time_stop;
    //double time_stamp[OPTIONS][NUM_TESTS];
    double final_answer;
    double time_stamp[OPTIONS][NUM_TESTS];
    final_answer = wakeup_delay();
    int NUM_THREADS = 2;
    // Read the matrix input data
    complexFloat *data;
    int height, width;
    // printf("fftLength = %d\n", fftLength);
    // printf("fftType = %d\n", fftType);

    FILE *fpfft = fopen("fft_perf_Thread_radix4.csv","w");


    complexFloat *fftInput = (complexFloat*) malloc(sizeof(complexFloat) * (int)pow(2, NUM_TESTS));
    complexFloat *spectrum = (complexFloat*) malloc(sizeof(complexFloat) * (int)pow(2, NUM_TESTS));


    int N = (int)pow(2, NUM_TESTS);
    float *signal = (float*) malloc(sizeof(float) * N);
    int frequencies[] = {4, 8, 18, 33, 152}; 
    int num_frequencies = sizeof(frequencies) / sizeof(frequencies[0]);

    for (int i = 0; i < N; i++) {
        signal[i] = 0.0;
        for (int j = 0; j < num_frequencies; j++) {
            signal[i] += cos(2 * M_PI * frequencies[j] * i / N);
        }
    }

    pthread_t threads[8];
    struct thread_data thread_data_array[8];
    pthread_barrier_t barrier;
    complexFloat *butterfly = malloc(sizeof(complexFloat) * (int)pow(2, NUM_TESTS));

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    int rc;
    long t;

    OPTION = 0;
    int fftRadix = 4;
    int fftStage = 0;
    int fftLength = (int)pow(fftRadix, fftStage);
    int fftType = DIT;
    int NUM_TESTS_1 = floor(NUM_TESTS / 2);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        fftLength = (int)pow(fftRadix, fftStage);  // <-- move this line BEFORE printf
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Serial\n", OPTION, fftRadix, fftStage, fftLength);
        for (i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        clock_gettime(CLOCK_REALTIME, &time_start);
        fft_radix4_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, DIT);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }

    OPTION++;
    NUM_THREADS = 2;
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }
    pthread_barrier_destroy(&barrier);
    OPTION++;
    NUM_THREADS = 3;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }
    pthread_barrier_destroy(&barrier);
    OPTION++;
    NUM_THREADS = 4;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }
    pthread_barrier_destroy(&barrier);
    OPTION++;
    NUM_THREADS = 5;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }
    pthread_barrier_destroy(&barrier);
    OPTION++;
    NUM_THREADS = 6;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }
    pthread_barrier_destroy(&barrier);
    OPTION++;
    NUM_THREADS = 7;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }
    pthread_barrier_destroy(&barrier);
    OPTION++;
    NUM_THREADS = 8;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (fftStage = 1; fftStage <= NUM_TESTS_1; fftStage++) {
        // Update fftLength and reassign input signal for the current stage
        fftLength = (int)pow(fftRadix, fftStage);
        width = fftLength;
        printf("OPTION %d: radix %d, fftStage=%d, fftLength=%d , DIT, Thread\n", OPTION, fftRadix, fftStage, fftLength);
        for (int i = 0; i < fftLength; i++) {
            fftInput[i].re = signal[i];
            fftInput[i].im = 0;
        }
        // Perform Radix-2 DIT FFT (Threaded)
        clock_gettime(CLOCK_REALTIME, &time_start);
        for (t = 0; t < NUM_THREADS; t++) {
            thread_data_array[t].thread_id = t;
            thread_data_array[t].num_threads = NUM_THREADS;
            thread_data_array[t].input = fftInput;
            thread_data_array[t].output = spectrum;
            thread_data_array[t].fftRadix = fftRadix;
            thread_data_array[t].fftStage = fftStage;
            thread_data_array[t].fftLength = fftLength;
            thread_data_array[t].butterfly = butterfly;
            thread_data_array[t].IFFT = false;
            thread_data_array[t].barrier = &barrier;
            rc = pthread_create(&threads[t], NULL, fft_radix4_DIT_thread, (void*) &thread_data_array[t]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for (t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][fftStage] = interval(time_start, time_stop);
    }


    pthread_barrier_destroy(&barrier);



    // Ensure all dynamically allocated memory is freed
    if (signal) free(signal);
    if (fftInput) free(fftInput);
    if (spectrum) free(spectrum);
    signal = NULL;
    fftInput = NULL;
    spectrum = NULL;
    long array_size = 0;
fprintf(fpfft, "Array_Size,Radix_2_DIT_Serial,Thread_2,Thread_3,Thread_4,Thread_5,Thread_6,Thread_7,Thread_8\n");
    for (int i = 1; i <= NUM_TESTS_1; i++) {
        array_size = pow(4, i);

        // Print nicely to terminal
        printf("Array Size: %ld | Serial: %f s | 2T: %f s | 3T: %f s | 4T: %f s | 5T: %f s | 6T: %f s | 7T: %f s | 8T: %f s\n",
            array_size,
            time_stamp[0][i], // Serial
            time_stamp[1][i], // 2T
            time_stamp[2][i], // 3T
            time_stamp[3][i], // 4T
            time_stamp[4][i], // 5T ✅
            time_stamp[5][i], // 6T ✅
            time_stamp[6][i], // 7T ✅
            time_stamp[7][i]  // 8T ✅
        );

        // Write to CSV file
             fprintf(fpfft, "%ld,%f,%f,%f,%f,%f,%f,%f,%f\n",
            array_size,
            time_stamp[0][i], // Serial
            time_stamp[1][i], // 2 threads
            time_stamp[2][i], // 3 threads
            time_stamp[3][i], // 4 threads
            time_stamp[4][i], // 5 threads  <<< you missed this one before
            time_stamp[5][i], // 6 threads
            time_stamp[6][i], // 7 threads
            time_stamp[7][i]  // 8 threads
            );
    
    }
    fclose(fpfft);


    return 0;
}
 
void* fft_radix2_DIF_thread(void *threadarg)
{
    struct thread_data *my_data = (struct thread_data *) threadarg;
    int thread_id = my_data->thread_id;
    int num_threads = my_data->num_threads;
    complexFloat *input = my_data->input;
    complexFloat *output = my_data->output;
    complexFloat *butterfly = my_data->butterfly;
    bool IFFT = my_data->IFFT;
    int fftRadix = my_data->fftRadix;  // Should be 2
    int fftStage = my_data->fftStage;
    int fftLength = my_data->fftLength;
    pthread_barrier_t *barrier = my_data->barrier;

    // Step 1: Initialize butterfly array (straight copy)
    long int low = (thread_id * fftLength) / num_threads;
    long int high = ((thread_id + 1) * fftLength) / num_threads;

    for (u32 i = low; i < high; i++) {
        if (IFFT) {
            butterfly[i].re = input[i].re;
            butterfly[i].im = -input[i].im;
        } else {
            butterfly[i] = input[i];
        }
    }

    pthread_barrier_wait(barrier); // Synchronize after copy

    // Step 2: FFT computation, stage-by-stage
    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = 1 << stage;                // 2^stage
        int groupSize = fftLength / (2 * groupNum);
        int total_butterflies = fftLength / 2;     // Always half of the array

        int butterflies_per_thread = (total_butterflies + num_threads - 1) / num_threads;
        int start = thread_id * butterflies_per_thread;
        int end = (start + butterflies_per_thread > total_butterflies) ? total_butterflies : start + butterflies_per_thread;

        for (int b = start; b < end; b++) {
            int g = b / groupSize;
            int k = b % groupSize;

            int idx0 = g * 2 * groupSize + k;
            int idx1 = idx0 + groupSize;
            int indexWeight = k * groupNum;

            complexFloat temp0 = complexAdd(butterfly[idx0], butterfly[idx1]);
            complexFloat temp1 = complexSub(butterfly[idx0], butterfly[idx1]);

            butterfly[idx0] = complexMul(temp0, getWeight(0, indexWeight, fftLength));
            butterfly[idx1] = complexMul(temp1, getWeight(1, indexWeight, fftLength));
        }

        pthread_barrier_wait(barrier); // Synchronize between stages
    }

    // Step 3: Bit-reversed output reordering
    for (u32 i = low; i < high; i++) {
        u32 iOrder = reverseBit(i, fftRadix, fftLength);
        if (IFFT) {
            output[iOrder].re = butterfly[i].re / fftLength;
            output[iOrder].im = -butterfly[i].im / fftLength;
        } else {
            output[iOrder] = butterfly[i];
        }
    }

    pthread_exit(NULL);
}
// Radix-2 FFT DIT (Threaded)
void* fft_radix2_DIT_thread(void *threadarg)
{
    struct thread_data *my_data = (struct thread_data *) threadarg;
    int thread_id = my_data->thread_id;
    int num_threads = my_data->num_threads;
    complexFloat *input = my_data->input;
    complexFloat *output = my_data->output;
    complexFloat *butterfly = my_data->butterfly;
    bool IFFT = my_data->IFFT;
    int fftRadix = my_data->fftRadix;  // Should be 2
    int fftStage = my_data->fftStage;
    int fftLength = my_data->fftLength;
    pthread_barrier_t *barrier = my_data->barrier;

    pthread_barrier_wait(barrier);

    long int low = (thread_id * fftLength) / num_threads;
    long int high = ((thread_id + 1) * fftLength) / num_threads;

    for (u32 i = low; i < high; i++) {
        u32 iOrder = reverseBit(i, fftRadix, fftLength);
        if (IFFT) {
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        } else {
            butterfly[iOrder] = input[i];
        }
    }

    pthread_barrier_wait(barrier);

    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = 1 << stage;                // 2^stage
        int groupSize = fftLength / (2 * groupNum);
        int total_butterflies = fftLength / 2;     // Always half of the array

        int butterflies_per_thread = (total_butterflies + num_threads - 1) / num_threads;
        int start = thread_id * butterflies_per_thread;
        int end = (start + butterflies_per_thread > total_butterflies) ? total_butterflies : start + butterflies_per_thread;

        for (int b = start; b < end; b++) {
            int g = b / groupSize;
            int k = b % groupSize;

            int idx0 = g * 2 * groupSize + k;
            int idx1 = idx0 + groupSize;
            int indexWeight = k * groupNum;

            complexFloat temp0 = complexAdd(butterfly[idx0], butterfly[idx1]);
            complexFloat temp1 = complexSub(butterfly[idx0], butterfly[idx1]);
            butterfly[idx0] = complexMul(temp0, getWeight(0, indexWeight, fftLength));
            butterfly[idx1] = complexMul(temp1, getWeight(1, indexWeight, fftLength));
        }
        pthread_barrier_wait(barrier); // Synchronize between stages
    }
    for (u32 i = low; i < high; i++) {
        u32 iOrder = reverseBit(i, fftRadix, fftLength);
        if (IFFT) {
            output[iOrder].re = butterfly[i].re / fftLength;
            output[iOrder].im = -butterfly[i].im / fftLength;
        } else {
            output[iOrder] = butterfly[i];
        }
    }
    pthread_exit(NULL);
}


void* fft_radix4_DIT_thread(void *threadarg)
{
    struct thread_data *my_data = (struct thread_data *) threadarg;
    int thread_id = my_data->thread_id;
    int num_threads = my_data->num_threads;
    complexFloat *input = my_data->input;
    complexFloat *output = my_data->output;
    complexFloat *butterfly = my_data->butterfly;
    bool IFFT = my_data->IFFT;
    int fftRadix = my_data->fftRadix;  // 4
    int fftStage = my_data->fftStage;
    int fftLength = my_data->fftLength;
    pthread_barrier_t *barrier = my_data->barrier;

    pthread_barrier_wait(barrier);

    long int low = (thread_id * fftLength) / num_threads;
    long int high = ((thread_id + 1) * fftLength) / num_threads;

    for (u32 i = low; i < high; i++) {
        u32 iOrder = reverseBit(i, fftRadix, fftLength);
        if (IFFT) {
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        } else {
            butterfly[iOrder] = input[i];
        }
    }

    pthread_barrier_wait(barrier);

    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = pow(4, fftStage - 1 - stage);
        int groupSize = pow(4, stage);
        int total_butterflies = fftLength / 4;

        int butterflies_per_thread = (total_butterflies + num_threads - 1) / num_threads;
        int start = thread_id * butterflies_per_thread;
        int end = (start + butterflies_per_thread > total_butterflies) ? total_butterflies : start + butterflies_per_thread;

        for (int b = start; b < end; b++) {
            int g = b / groupSize;
            int k = b % groupSize;

            int idx[4];
            for (int m = 0; m < 4; m++) {
                idx[m] = g * 4 * groupSize + k + m * groupSize;
            }
            int indexWeight = k * groupNum;

            complexFloat t0 = complexMul(butterfly[idx[0]], getWeight(0, indexWeight, fftLength));
            complexFloat t1 = complexMul(butterfly[idx[1]], getWeight(1, indexWeight, fftLength));
            complexFloat t2 = complexMul(butterfly[idx[2]], getWeight(2, indexWeight, fftLength));
            complexFloat t3 = complexMul(butterfly[idx[3]], getWeight(3, indexWeight, fftLength));

            complexFloat a0 = complexAdd(t0, t2);
            complexFloat a1 = complexAdd(t1, t3);
            complexFloat a2 = complexSub(t0, t2);
            complexFloat a3 = complexSub(t1, t3);
            a3 = complexMul(a3, (complexFloat){0, -1});

            butterfly[idx[0]] = complexAdd(a0, a1);
            butterfly[idx[1]] = complexAdd(a2, a3);
            butterfly[idx[2]] = complexSub(a0, a1);
            butterfly[idx[3]] = complexSub(a2, a3);
        }
        pthread_barrier_wait(barrier);
    }

    for (u32 i = low; i < high; i++) {
        if (IFFT) {
            output[i].re = butterfly[i].re / fftLength;
            output[i].im = -butterfly[i].im / fftLength;
        } else {
            output[i] = butterfly[i];
        }
    }

    pthread_exit(NULL);
}


void* fft_radix8_DIT_thread(void *threadarg)
{
    struct thread_data *my_data = (struct thread_data *) threadarg;
    int thread_id = my_data->thread_id;
    int num_threads = my_data->num_threads;
    complexFloat *input = my_data->input;
    complexFloat *output = my_data->output;
    complexFloat *butterfly = my_data->butterfly;
    bool IFFT = my_data->IFFT;
    int fftRadix = my_data->fftRadix;  // 8
    int fftStage = my_data->fftStage;
    int fftLength = my_data->fftLength;
    pthread_barrier_t *barrier = my_data->barrier;

    pthread_barrier_wait(barrier);

    long int low = (thread_id * fftLength) / num_threads;
    long int high = ((thread_id + 1) * fftLength) / num_threads;

    for (u32 i = low; i < high; i++) {
        u32 iOrder = reverseBit(i, fftRadix, fftLength);
        if (IFFT) {
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        } else {
            butterfly[iOrder] = input[i];
        }
    }

    pthread_barrier_wait(barrier);

    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = pow(fftRadix, fftStage - 1 - stage);
        int groupSize = pow(fftRadix, stage);
        int total_butterflies = fftLength / 8;

        int butterflies_per_thread = (total_butterflies + num_threads - 1) / num_threads;
        int start = thread_id * butterflies_per_thread;
        int end = (start + butterflies_per_thread > total_butterflies) ? total_butterflies : start + butterflies_per_thread;

        for (int b = start; b < end; b++) {
            int g = b / groupSize;
            int k = b % groupSize;

            int idx[8];
            for (int m = 0; m < 8; m++) {
                idx[m] = g * 8 * groupSize + k + m * groupSize;
            }
            int indexWeight = k * groupNum;

            complexFloat t0 = complexMul(butterfly[idx[0]], getWeight(0, indexWeight, fftLength));
            complexFloat t1 = complexMul(butterfly[idx[1]], getWeight(1, indexWeight, fftLength));
            complexFloat t2 = complexMul(butterfly[idx[2]], getWeight(2, indexWeight, fftLength));
            complexFloat t3 = complexMul(butterfly[idx[3]], getWeight(3, indexWeight, fftLength));
            complexFloat t4 = complexMul(butterfly[idx[4]], getWeight(4, indexWeight, fftLength));
            complexFloat t5 = complexMul(butterfly[idx[5]], getWeight(5, indexWeight, fftLength));
            complexFloat t6 = complexMul(butterfly[idx[6]], getWeight(6, indexWeight, fftLength));
            complexFloat t7 = complexMul(butterfly[idx[7]], getWeight(7, indexWeight, fftLength));

            complexFloat a0 = complexAdd(t0, t4);
            complexFloat a1 = complexAdd(t2, t6);
            complexFloat a2 = complexAdd(t1, t5);
            complexFloat a3 = complexAdd(t3, t7);
            complexFloat a4 = complexSub(t0, t4);
            complexFloat a5 = complexSub(t2, t6);
            complexFloat a6 = complexSub(t1, t5);
            complexFloat a7 = complexSub(t3, t7);

            a5 = complexMul(a5, (complexFloat){0, -1});
            a7 = complexMul(a7, (complexFloat){0, -1});

            butterfly[idx[0]] = complexAdd(a0, a1);
            butterfly[idx[1]] = complexAdd(a4, a5);
            butterfly[idx[2]] = complexAdd(a2, a3);
            butterfly[idx[3]] = complexAdd(a6, a7);
            butterfly[idx[4]] = complexSub(a0, a1);
            butterfly[idx[5]] = complexSub(a4, a5);
            butterfly[idx[6]] = complexSub(a2, a3);
            butterfly[idx[7]] = complexSub(a6, a7);
        }
        pthread_barrier_wait(barrier);
    }

    for (u32 i = low; i < high; i++) {
        if (IFFT) {
            output[i].re = butterfly[i].re / fftLength;
            output[i].im = -butterfly[i].im / fftLength;
        } else {
            output[i] = butterfly[i];
        }
    }

    pthread_exit(NULL);
}

static u32 reverseBit(
    u32 n,
    int fftRadix,
    int fftLength
){
    u32 bitWidth = 0;
    while((1<<bitWidth) < fftLength) bitWidth++;
    if(fftRadix == 8){
        n = ((n & 0b111000111000111000111000) >> 3 ) | ((n & 0b000111000111000111000111) << 3 );
        n = ((n & 0b111111000000111111000000) >> 6 ) | ((n & 0b000000111111000000111111) << 6 );
        n = ((n & 0b111111111111000000000000) >> 12 ) | ((n & 0b000000000000111111111111) << 12 );
        n = n >> (24 - bitWidth);
    }else{
        if(fftRadix == 2) n = ((n & 0xAAAAAAAA) >> 1 ) | ((n & 0x55555555) << 1 );
        n = ((n & 0xCCCCCCCC) >> 2 ) | ((n & 0x33333333) << 2 );
        n = ((n & 0xF0F0F0F0) >> 4 ) | ((n & 0x0F0F0F0F) << 4 );
        n = ((n & 0xFF00FF00) >> 8 ) | ((n & 0x00FF00FF) << 8 );
        n = ((n & 0xFFFF0000) >> 16 ) | ((n & 0x0000FFFF) << 16 );
        n = n >> (32 - bitWidth);
    }
    return n;
}



static complexFloat getWeight(
    int iButterfly,
    int indexWeight,
    int fftLength
){
    complexFloat result;
    result.re = cos(2 * M_PI * iButterfly * indexWeight / fftLength);
    result.im = -sin(2 * M_PI * iButterfly * indexWeight / fftLength);
    return result;
}


static complexFloat complexAdd(
    complexFloat A,
    complexFloat B
){
    complexFloat result;
    result.re = A.re + B.re;
    result.im = A.im + B.im;
    return result;
}

static complexFloat complexSub(
    complexFloat A,
    complexFloat B
){
    complexFloat result;
    result.re = A.re - B.re;
    result.im = A.im - B.im;
    return result;
}

static complexFloat complexMul(
    complexFloat A,
    complexFloat B
){
    complexFloat result;
    result.re = A.re * B.re - A.im * B.im;
    result.im = A.im * B.re + A.re * B.im;
    return result;
}



// Radix-2 FFT DIT
int fft_radix2_DIT(complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType ){
    complexFloat *butterfly = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  reverseBit(i,fftRadix,fftLength);
        if(IFFT){
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        }else{
            butterfly[iOrder] = input[i];
        }
    }  
    complexFloat tmpButterfly[(int)(log(fftRadix)/log(2))][fftRadix];
    int indexButterfly[fftRadix];
    int indexWeight;
    for(int i = 0; i < fftStage; i++){
        int groupNum, groupSize;
        groupNum =  pow(fftRadix, fftStage-1 - i) ;
        groupSize =  pow(fftRadix, i) ;
        for(int j = 0; j < groupNum; j++){
            for(int k = 0; k < groupSize; k++){
                for(int m = 0; m < fftRadix; m++){
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;
                tmpButterfly[0][0] = complexMul(butterfly[indexButterfly[0]], getWeight(0,indexWeight, fftLength));
                tmpButterfly[0][1] = complexMul(butterfly[indexButterfly[1]], getWeight(1,indexWeight, fftLength));
                butterfly[indexButterfly[0]] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][1]);
                butterfly[indexButterfly[1]] = complexSub(tmpButterfly[0][0], tmpButterfly[0][1]);
                    
            }
        }
    }

    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  i;
        if(IFFT){
            output[iOrder].re = butterfly[i].re / fftLength;
    	    output[iOrder].im = -butterfly[i].im / fftLength;
        }else{
            output[iOrder] = butterfly[i];
        }
    }
    free(butterfly);
    butterfly = NULL;
    return 0;
}

// Radix-4 FFT DIT
int fft_radix4_DIT( complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType ){
    complexFloat *butterfly = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder = (fftType == DIT) ? reverseBit(i,fftRadix,fftLength) : i;
        if(IFFT){
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        }else{
            butterfly[iOrder] = input[i];
        }
    }
    complexFloat tmpButterfly[(int)(log(fftRadix)/log(2))][fftRadix];
    int indexButterfly[fftRadix];
    int indexWeight;
    for(int i = 0; i < fftStage; i++){
        int groupNum, groupSize;
        groupNum =  pow(fftRadix, fftStage-1 - i) ;
        groupSize =  pow(fftRadix, i) ;
        for(int j = 0; j < groupNum; j++){
            for(int k = 0; k < groupSize; k++){
                for(int m = 0; m < fftRadix; m++){
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;
                tmpButterfly[0][0] = complexMul(butterfly[indexButterfly[0]], getWeight(0,indexWeight, fftLength));
                tmpButterfly[0][1] = complexMul(butterfly[indexButterfly[1]], getWeight(1,indexWeight, fftLength));
                tmpButterfly[0][2] = complexMul(butterfly[indexButterfly[2]], getWeight(2,indexWeight, fftLength));
                tmpButterfly[0][3] = complexMul(butterfly[indexButterfly[3]], getWeight(3,indexWeight, fftLength));
                tmpButterfly[1][0] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][2]);
                tmpButterfly[1][2] = complexSub(tmpButterfly[0][0], tmpButterfly[0][2]);
                tmpButterfly[1][1] = complexAdd(tmpButterfly[0][1], tmpButterfly[0][3]);
                tmpButterfly[1][3] = complexSub(tmpButterfly[0][1], tmpButterfly[0][3]);
                tmpButterfly[1][3] = complexMul(tmpButterfly[1][3], (complexFloat){0,-1});
                butterfly[indexButterfly[0]] = complexAdd(tmpButterfly[1][0], tmpButterfly[1][1]);
                butterfly[indexButterfly[1]] = complexAdd(tmpButterfly[1][2], tmpButterfly[1][3]);
                butterfly[indexButterfly[2]] = complexSub(tmpButterfly[1][0], tmpButterfly[1][1]);
                butterfly[indexButterfly[3]] = complexSub(tmpButterfly[1][2], tmpButterfly[1][3]);
                   
            }
        }
    }

    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  i;
        if(IFFT){
            output[iOrder].re = butterfly[i].re / fftLength;
    	    output[iOrder].im = -butterfly[i].im / fftLength;
        }else{
            output[iOrder] = butterfly[i];
        }
    }
    free(butterfly);
    butterfly = NULL;
    return 0;
}


// Radix-8 FFT DIT
int fft_radix8_DIT(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType ){
    complexFloat *butterfly = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  reverseBit(i,fftRadix,fftLength) ;
        if(IFFT){
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        }else{
            butterfly[iOrder] = input[i];
        }
    }
    
    complexFloat tmpButterfly[(int)(log(fftRadix)/log(2))][fftRadix];
    int indexButterfly[fftRadix];
    int indexWeight;

    for(int i = 0; i < fftStage; i++){
        int groupNum, groupSize;
        groupNum = pow(fftRadix, fftStage-1 - i) ;
        groupSize =  pow(fftRadix, i) ;
        for(int j = 0; j < groupNum; j++){
            for(int k = 0; k < groupSize; k++){
                for(int m = 0; m < fftRadix; m++){
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;
                tmpButterfly[0][0] = complexMul(butterfly[indexButterfly[0]], getWeight(0,indexWeight, fftLength));
                tmpButterfly[0][1] = complexMul(butterfly[indexButterfly[1]], getWeight(1,indexWeight, fftLength));
                tmpButterfly[0][2] = complexMul(butterfly[indexButterfly[2]], getWeight(2,indexWeight, fftLength));
                tmpButterfly[0][3] = complexMul(butterfly[indexButterfly[3]], getWeight(3,indexWeight, fftLength));
                tmpButterfly[0][4] = complexMul(butterfly[indexButterfly[4]], getWeight(4,indexWeight, fftLength));
                tmpButterfly[0][5] = complexMul(butterfly[indexButterfly[5]], getWeight(5,indexWeight, fftLength));
                tmpButterfly[0][6] = complexMul(butterfly[indexButterfly[6]], getWeight(6,indexWeight, fftLength));
                tmpButterfly[0][7] = complexMul(butterfly[indexButterfly[7]], getWeight(7,indexWeight, fftLength));
                tmpButterfly[1][0] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][4]);
                tmpButterfly[1][4] = complexSub(tmpButterfly[0][0], tmpButterfly[0][4]);
                tmpButterfly[1][2] = complexAdd(tmpButterfly[0][2], tmpButterfly[0][6]);
                tmpButterfly[1][6] = complexSub(tmpButterfly[0][2], tmpButterfly[0][6]);
                tmpButterfly[1][1] = complexAdd(tmpButterfly[0][1], tmpButterfly[0][5]);
                tmpButterfly[1][5] = complexSub(tmpButterfly[0][1], tmpButterfly[0][5]);
                tmpButterfly[1][3] = complexAdd(tmpButterfly[0][3], tmpButterfly[0][7]);
                tmpButterfly[1][7] = complexSub(tmpButterfly[0][3], tmpButterfly[0][7]);
                tmpButterfly[1][6] = complexMul(tmpButterfly[1][6], (complexFloat){0,-1});
                tmpButterfly[1][7] = complexMul(tmpButterfly[1][7], (complexFloat){0,-1});
                tmpButterfly[2][0] = complexAdd(tmpButterfly[1][0], tmpButterfly[1][2]);
                tmpButterfly[2][2] = complexSub(tmpButterfly[1][0], tmpButterfly[1][2]);
                tmpButterfly[2][4] = complexAdd(tmpButterfly[1][4], tmpButterfly[1][6]);
                tmpButterfly[2][6] = complexSub(tmpButterfly[1][4], tmpButterfly[1][6]);
                tmpButterfly[2][1] = complexAdd(tmpButterfly[1][1], tmpButterfly[1][3]);
                tmpButterfly[2][3] = complexSub(tmpButterfly[1][1], tmpButterfly[1][3]);
                tmpButterfly[2][5] = complexAdd(tmpButterfly[1][5], tmpButterfly[1][7]);
                tmpButterfly[2][7] = complexSub(tmpButterfly[1][5], tmpButterfly[1][7]);
                tmpButterfly[2][5] = complexMul(tmpButterfly[2][5], (complexFloat){cos(M_PI/4),-sin(M_PI/4)});
                tmpButterfly[2][3] = complexMul(tmpButterfly[2][3], (complexFloat){0,-1});
                tmpButterfly[2][7] = complexMul(tmpButterfly[2][7], (complexFloat){cos(3*M_PI/4),-sin(3*M_PI/4)});
                butterfly[indexButterfly[0]] = complexAdd(tmpButterfly[2][0], tmpButterfly[2][1]);
                butterfly[indexButterfly[1]] = complexAdd(tmpButterfly[2][4], tmpButterfly[2][5]);
                butterfly[indexButterfly[2]] = complexAdd(tmpButterfly[2][2], tmpButterfly[2][3]);
                butterfly[indexButterfly[3]] = complexAdd(tmpButterfly[2][6], tmpButterfly[2][7]);
                butterfly[indexButterfly[4]] = complexSub(tmpButterfly[2][0], tmpButterfly[2][1]);
                butterfly[indexButterfly[5]] = complexSub(tmpButterfly[2][4], tmpButterfly[2][5]);
                butterfly[indexButterfly[6]] = complexSub(tmpButterfly[2][2], tmpButterfly[2][3]);
                butterfly[indexButterfly[7]] = complexSub(tmpButterfly[2][6], tmpButterfly[2][7]);
            }
            
        }
    }

    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  i;
        if(IFFT){
            output[iOrder].re = butterfly[i].re / fftLength;
    	    output[iOrder].im = -butterfly[i].im / fftLength;
        }else{
            output[iOrder] = butterfly[i];
        }
    }
    free(butterfly);
    butterfly = NULL;
    return 0;
}