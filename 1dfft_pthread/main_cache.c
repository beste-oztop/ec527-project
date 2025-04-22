/* perf stat -e cache-references,cache-misses ./fft_main 2 12 1 
gcc -pthread main.c -lpthread -lm -lrt -o fft_main
OMP_NUM_THREADS=8 ./fft_main fft_input.txt 2 12 1
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

typedef unsigned int u32;
typedef unsigned short u16;
typedef char u8;
#define DIT 0 
#define DIF 1 

typedef struct _complexFloat{
    float re;
    float im;
}complexFloat;

#define NUM_THREADS 4   /* Ensure this matches the number of threads used in pthread_barrier_init */

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

void* fft_radix2_DIF(void *threadarg);
void* fft_radix2_DIT(void *threadarg);
void* fft_radix4_DIT(void *threadarg);
void* fft_radix4_DIF(void *threadarg);
void* fft_radix8_DIT(void *threadarg);
void* fft_radix8_DIF(void *threadarg);
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
    int i;
    struct timespec time_start, time_stop;
    //double time_stamp[OPTIONS][NUM_TESTS];
    double final_answer;
    int ognt = 0;
    if (argc != 5) {
        printf("Usage: %s <input_file.txt> <fftRadix: 2|4|8> <fftStage> <fftType: 0 (DIT) | 1 (DIF)>\n", argv[0]);
        return 1;
    }

    final_answer = wakeup_delay();

    // Read the matrix input data
    complexFloat *data;
    int height, width;
    int fftRadix = atoi(argv[2]);
    int fftStage = atoi(argv[3]);
    int fftType = atoi(argv[4]);
    int fftLength = (int)pow(fftRadix, fftStage);
    width=fftLength;
    printf("fftLength = %d\n", fftLength);
    printf("fftType = %d\n", fftType);

    FILE *fpsignal = fopen("cossignal.txt","r");
    FILE *fpfftInput = fopen("1fftInput.txt","w");
    FILE *fpspectrum = fopen("2spectrum.txt","w");
    FILE *fpifftOutput = fopen("3ifftOutput.txt","w");
    if (fpsignal == NULL ||
        fpfftInput == NULL ||
        fpspectrum == NULL ||
        fpifftOutput == NULL){
        printf("Error: open file failed!\n");
        return 1;
    }

    float *signal = (float*) malloc(sizeof(float) * fftLength);
    complexFloat *fftInput = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    complexFloat *spectrum = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    complexFloat *ifftOutput = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);

    for(int i = 0; i < fftLength; i++){
        fscanf(fpsignal,"%f", &signal[i]);
    }
    fclose(fpsignal);

    for(int i = 0; i < fftLength; i++){
        fftInput[i].re = signal[i];
        fftInput[i].im = 0;
        fprintf(fpfftInput, "%f\t%f\n", fftInput[i].re, fftInput[i].im);
    }
    fclose(fpfftInput);
    printf("read input signal data finished !\n");
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    pthread_barrier_t barrier;
    complexFloat *butterfly = malloc(sizeof(complexFloat) * fftLength);

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    int rc;
    long t;
      // Switch case between different FFT types
    switch (fftRadix){
        case 2:
            if(fftType == DIT){
                // --- Start perf counters for Radix2 DIT ---
                struct perf_event_attr pe_refs, pe_misses;
                long fd_refs, fd_misses;
                memset(&pe_refs, 0, sizeof(struct perf_event_attr));
                pe_refs.type = PERF_TYPE_HARDWARE;
                pe_refs.size = sizeof(struct perf_event_attr);
                pe_refs.config = PERF_COUNT_HW_CACHE_REFERENCES;
                pe_refs.disabled = 1;
                pe_refs.exclude_kernel = 1;
                pe_refs.exclude_hv = 1;
                
                memset(&pe_misses, 0, sizeof(struct perf_event_attr));
                pe_misses.type = PERF_TYPE_HARDWARE;
                pe_misses.size = sizeof(struct perf_event_attr);
                pe_misses.config = PERF_COUNT_HW_CACHE_MISSES;
                pe_misses.disabled = 1;
                pe_misses.exclude_kernel = 1;
                pe_misses.exclude_hv = 1;
                
                fd_refs = perf_event_open(&pe_refs, 0, -1, -1, 0);
                if(fd_refs == -1) { perror("perf_event_open for cache references"); }
                fd_misses = perf_event_open(&pe_misses, 0, -1, -1, 0);
                if(fd_misses == -1) { perror("perf_event_open for cache misses"); }
                
                ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
                // --- End start perf counters ---
                
                clock_gettime(CLOCK_REALTIME, &time_start);
                for (t = 0; t < NUM_THREADS; t++) {
                    thread_data_array[t].thread_id = t;
                    thread_data_array[t].num_threads = NUM_THREADS;
                    thread_data_array[t].input = fftInput;
                    thread_data_array[t].output = spectrum;
                    thread_data_array[t].fftRadix = fftRadix;
                    thread_data_array[t].fftStage = fftStage;
                    thread_data_array[t].butterfly = butterfly;
                    thread_data_array[t].fftLength = fftLength;
                    thread_data_array[t].IFFT = false;
                    thread_data_array[t].barrier = &barrier;
                    rc = pthread_create(&threads[t], NULL, fft_radix2_DIT, (void*) &thread_data_array[t]);
                    if (rc) {
                        printf("ERROR; return code from pthread_create() is %d\n", rc);
                        exit(-1);
                    }
                }
                for (t = 0; t < NUM_THREADS; t++) {
                    pthread_join(threads[t], NULL);
                }
                clock_gettime(CLOCK_REALTIME, &time_stop);
                
                // --- Stop and read perf counters ---
                ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
                long long count_refs = 0, count_misses = 0;
                read(fd_refs, &count_refs, sizeof(long long));
                read(fd_misses, &count_misses, sizeof(long long));
                printf("Radix2 DIT - Cache references: %lld\n", count_refs);
                printf("Radix2 DIT - Cache misses: %lld\n", count_misses);
                close(fd_refs);
                close(fd_misses);
                // --- End perf counters ---
                
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix2 DIT: %f seconds\n", time_fft);
                FILE *outputFile = fopen("1DFFT_Radix2_output_c_DIT.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < width; i++) {
                    fprintf(outputFile, "%f %f\n", spectrum[i].re, spectrum[i].im);
                }
                fclose(outputFile);
            } else {  // Radix2 DIF with perf counters added
                // --- Start perf counters for Radix2 DIF ---
                struct perf_event_attr pe_refs, pe_misses;
                long fd_refs, fd_misses;
                memset(&pe_refs, 0, sizeof(struct perf_event_attr));
                pe_refs.type = PERF_TYPE_HARDWARE;
                pe_refs.size = sizeof(struct perf_event_attr);
                pe_refs.config = PERF_COUNT_HW_CACHE_REFERENCES;
                pe_refs.disabled = 1;
                pe_refs.exclude_kernel = 1;
                pe_refs.exclude_hv = 1;
                
                memset(&pe_misses, 0, sizeof(struct perf_event_attr));
                pe_misses.type = PERF_TYPE_HARDWARE;
                pe_misses.size = sizeof(struct perf_event_attr);
                pe_misses.config = PERF_COUNT_HW_CACHE_MISSES;
                pe_misses.disabled = 1;
                pe_misses.exclude_kernel = 1;
                pe_misses.exclude_hv = 1;
                
                fd_refs = perf_event_open(&pe_refs, 0, -1, -1, 0);
                if(fd_refs == -1) { perror("perf_event_open for cache references"); }
                fd_misses = perf_event_open(&pe_misses, 0, -1, -1, 0);
                if(fd_misses == -1) { perror("perf_event_open for cache misses"); }
                
                ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
                // --- End start perf counters ---
                
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
                    rc = pthread_create(&threads[t], NULL, fft_radix2_DIF, (void*) &thread_data_array[t]);
                    if (rc) {
                        printf("ERROR; return code from pthread_create() is %d\n", rc);
                        exit(-1);
                    }
                }
                for (t = 0; t < NUM_THREADS; t++) {
                    pthread_join(threads[t], NULL);
                }
                clock_gettime(CLOCK_REALTIME, &time_stop);
                
                // --- Stop and read perf counters ---
                ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
                long long count_refs = 0, count_misses = 0;
                read(fd_refs, &count_refs, sizeof(long long));
                read(fd_misses, &count_misses, sizeof(long long));
                printf("Radix2 DIF - Cache references: %lld\n", count_refs);
                printf("Radix2 DIF - Cache misses: %lld\n", count_misses);
                close(fd_refs);
                close(fd_misses);
                // --- End perf counters ---
                
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix2 DIF: %f seconds\n", time_fft);
                FILE *outputFile = fopen("1DFFT_Radix2_output_c_DIF.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < width; i++) {
                    fprintf(outputFile, "%f %f\n", spectrum[i].re, spectrum[i].im);
                }
                fclose(outputFile);
            }
            break;
        case 4:
            if(fftType == DIT){
                // --- Start perf counters for Radix4 DIT ---
                struct perf_event_attr pe_refs, pe_misses;
                long fd_refs, fd_misses;
                memset(&pe_refs, 0, sizeof(struct perf_event_attr));
                pe_refs.type = PERF_TYPE_HARDWARE;
                pe_refs.size = sizeof(struct perf_event_attr);
                pe_refs.config = PERF_COUNT_HW_CACHE_REFERENCES;
                pe_refs.disabled = 1;
                pe_refs.exclude_kernel = 1;
                pe_refs.exclude_hv = 1;
                
                memset(&pe_misses, 0, sizeof(struct perf_event_attr));
                pe_misses.type = PERF_TYPE_HARDWARE;
                pe_misses.size = sizeof(struct perf_event_attr);
                pe_misses.config = PERF_COUNT_HW_CACHE_MISSES;
                pe_misses.disabled = 1;
                pe_misses.exclude_kernel = 1;
                pe_misses.exclude_hv = 1;
                
                fd_refs = perf_event_open(&pe_refs, 0, -1, -1, 0);
                if(fd_refs == -1) { perror("perf_event_open for cache references"); }
                fd_misses = perf_event_open(&pe_misses, 0, -1, -1, 0);
                if(fd_misses == -1) { perror("perf_event_open for cache misses"); }
                
                ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
                // --- End start perf counters ---
                
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
                    rc = pthread_create(&threads[t], NULL, fft_radix4_DIT, (void*) &thread_data_array[t]);
                    if (rc) {
                        printf("ERROR; return code from pthread_create() is %d\n", rc);
                        exit(-1);
                    }
                }
                for (t = 0; t < NUM_THREADS; t++) {
                    pthread_join(threads[t], NULL);
                }
                clock_gettime(CLOCK_REALTIME, &time_stop);
                // --- Stop and read perf counters ---
                ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
                {
                    long long count_refs = 0, count_misses = 0;
                    read(fd_refs, &count_refs, sizeof(long long));
                    read(fd_misses, &count_misses, sizeof(long long));
                    printf("Radix4 DIT - Cache references: %lld\n", count_refs);
                    printf("Radix4 DIT - Cache misses: %lld\n", count_misses);
                }
                close(fd_refs);
                close(fd_misses);
                // --- End perf counters ---
                
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix4 DIT: %f seconds\n", time_fft);
                FILE *outputFile = fopen("1DFFT_Radix4_output_c_DIT.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < width; i++) {
                    fprintf(outputFile, "%f %f\n", spectrum[i].re, spectrum[i].im);
                }
                fclose(outputFile);
            } else {  // Radix4 DIF with perf counters
                // --- Start perf counters for Radix4 DIF ---
                struct perf_event_attr pe_refs, pe_misses;
                long fd_refs, fd_misses;
                memset(&pe_refs, 0, sizeof(struct perf_event_attr));
                pe_refs.type = PERF_TYPE_HARDWARE;
                pe_refs.size = sizeof(struct perf_event_attr);
                pe_refs.config = PERF_COUNT_HW_CACHE_REFERENCES;
                pe_refs.disabled = 1;
                pe_refs.exclude_kernel = 1;
                pe_refs.exclude_hv = 1;
                
                memset(&pe_misses, 0, sizeof(struct perf_event_attr));
                pe_misses.type = PERF_TYPE_HARDWARE;
                pe_misses.size = sizeof(struct perf_event_attr);
                pe_misses.config = PERF_COUNT_HW_CACHE_MISSES;
                pe_misses.disabled = 1;
                pe_misses.exclude_kernel = 1;
                pe_misses.exclude_hv = 1;
                
                fd_refs = perf_event_open(&pe_refs, 0, -1, -1, 0);
                if(fd_refs == -1) { perror("perf_event_open for cache references"); }
                fd_misses = perf_event_open(&pe_misses, 0, -1, -1, 0);
                if(fd_misses == -1) { perror("perf_event_open for cache misses"); }
                
                ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
                // --- End start perf counters ---
                
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
                    rc = pthread_create(&threads[t], NULL, fft_radix4_DIF, (void*) &thread_data_array[t]);
                    if (rc) {
                        printf("ERROR; return code from pthread_create() is %d\n", rc);
                        exit(-1);
                    }
                }
                for (t = 0; t < NUM_THREADS; t++) {
                    pthread_join(threads[t], NULL);
                }
                clock_gettime(CLOCK_REALTIME, &time_stop);
                // --- Stop and read perf counters ---
                ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
                {
                    long long count_refs = 0, count_misses = 0;
                    read(fd_refs, &count_refs, sizeof(long long));
                    read(fd_misses, &count_misses, sizeof(long long));
                    printf("Radix4 DIF - Cache references: %lld\n", count_refs);
                    printf("Radix4 DIF - Cache misses: %lld\n", count_misses);
                }
                close(fd_refs);
                close(fd_misses);
                // --- End perf counters ---
                
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix4 DIF: %f seconds\n", time_fft);
                FILE *outputFile = fopen("1DFFT_Radix4_output_c_DIF.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < width; i++) {
                    fprintf(outputFile, "%f %f\n", spectrum[i].re, spectrum[i].im);
                }
                fclose(outputFile);
            }
            break;
        case 8:
            if(fftType == DIT){
                // --- Start perf counters for Radix8 DIT ---
                struct perf_event_attr pe_refs, pe_misses;
                long fd_refs, fd_misses;
                memset(&pe_refs, 0, sizeof(struct perf_event_attr));
                pe_refs.type = PERF_TYPE_HARDWARE;
                pe_refs.size = sizeof(struct perf_event_attr);
                pe_refs.config = PERF_COUNT_HW_CACHE_REFERENCES;
                pe_refs.disabled = 1;
                pe_refs.exclude_kernel = 1;
                pe_refs.exclude_hv = 1;
                
                memset(&pe_misses, 0, sizeof(struct perf_event_attr));
                pe_misses.type = PERF_TYPE_HARDWARE;
                pe_misses.size = sizeof(struct perf_event_attr);
                pe_misses.config = PERF_COUNT_HW_CACHE_MISSES;
                pe_misses.disabled = 1;
                pe_misses.exclude_kernel = 1;
                pe_misses.exclude_hv = 1;
                
                fd_refs = perf_event_open(&pe_refs, 0, -1, -1, 0);
                if(fd_refs == -1) { perror("perf_event_open for cache references"); }
                fd_misses = perf_event_open(&pe_misses, 0, -1, -1, 0);
                if(fd_misses == -1) { perror("perf_event_open for cache misses"); }
                
                ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
                // --- End start perf counters ---
                
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
                    rc = pthread_create(&threads[t], NULL, fft_radix8_DIT, (void*) &thread_data_array[t]);
                    if (rc) {
                        printf("ERROR; return code from pthread_create() is %d\n", rc);
                        exit(-1);
                    }
                }
                for (t = 0; t < NUM_THREADS; t++) {
                    pthread_join(threads[t], NULL);
                }
                clock_gettime(CLOCK_REALTIME, &time_stop);
                
                // --- Stop and read perf counters ---
                ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
                {
                    long long count_refs = 0, count_misses = 0;
                    read(fd_refs, &count_refs, sizeof(long long));
                    read(fd_misses, &count_misses, sizeof(long long));
                    printf("Radix8 DIT - Cache references: %lld\n", count_refs);
                    printf("Radix8 DIT - Cache misses: %lld\n", count_misses);
                }
                close(fd_refs);
                close(fd_misses);
                // --- End perf counters ---
                
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix8 DIT: %f seconds\n", time_fft);
                FILE *outputFile = fopen("1DFFT_Radix8_output_c_DIT.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < width; i++) {
                    fprintf(outputFile, "%f %f\n", spectrum[i].re, spectrum[i].im);
                }
                fclose(outputFile);
            } else {  // Radix8 DIF with perf counters
                // --- Start perf counters for Radix8 DIF ---
                struct perf_event_attr pe_refs, pe_misses;
                long fd_refs, fd_misses;
                memset(&pe_refs, 0, sizeof(struct perf_event_attr));
                pe_refs.type = PERF_TYPE_HARDWARE;
                pe_refs.size = sizeof(struct perf_event_attr);
                pe_refs.config = PERF_COUNT_HW_CACHE_REFERENCES;
                pe_refs.disabled = 1;
                pe_refs.exclude_kernel = 1;
                pe_refs.exclude_hv = 1;
                
                memset(&pe_misses, 0, sizeof(struct perf_event_attr));
                pe_misses.type = PERF_TYPE_HARDWARE;
                pe_misses.size = sizeof(struct perf_event_attr);
                pe_misses.config = PERF_COUNT_HW_CACHE_MISSES;
                pe_misses.disabled = 1;
                pe_misses.exclude_kernel = 1;
                pe_misses.exclude_hv = 1;
                
                fd_refs = perf_event_open(&pe_refs, 0, -1, -1, 0);
                if(fd_refs == -1) { perror("perf_event_open for cache references"); }
                fd_misses = perf_event_open(&pe_misses, 0, -1, -1, 0);
                if(fd_misses == -1) { perror("perf_event_open for cache misses"); }
                
                ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
                // --- End start perf counters ---
                
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
                    rc = pthread_create(&threads[t], NULL, fft_radix8_DIF, (void*) &thread_data_array[t]);
                    if (rc) {
                        printf("ERROR; return code from pthread_create() is %d\n", rc);
                        exit(-1);
                    }
                }
                for (t = 0; t < NUM_THREADS; t++) {
                    pthread_join(threads[t], NULL);
                }
                clock_gettime(CLOCK_REALTIME, &time_stop);
                
                // --- Stop and read perf counters ---
                ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
                {
                    long long count_refs = 0, count_misses = 0;
                    read(fd_refs, &count_refs, sizeof(long long));
                    read(fd_misses, &count_misses, sizeof(long long));
                    printf("Radix8 DIF - Cache references: %lld\n", count_refs);
                    printf("Radix8 DIF - Cache misses: %lld\n", count_misses);
                }
                close(fd_refs);
                close(fd_misses);
                // --- End perf counters ---
                
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix8 DIF: %f seconds\n", time_fft);
                FILE *outputFile = fopen("1DFFT_Radix8_output_c_DIF.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < width; i++) {
                    fprintf(outputFile, "%f %f\n", spectrum[i].re, spectrum[i].im);
                }
                fclose(outputFile);
            }
            break;
    }
    pthread_barrier_destroy(&barrier);
    
    // Ensure all dynamically allocated memory is freed
    if (signal) free(signal);
    if (fftInput) free(fftInput);
    if (spectrum) free(spectrum);
    if (ifftOutput) free(ifftOutput);
    signal = NULL;
    fftInput = NULL;
    spectrum = NULL;
    ifftOutput = NULL;
    return 0;
}
 
void* fft_radix2_DIF(void *threadarg)
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

        int total_groups = groupNum;
        int groups_per_thread = (total_groups + num_threads - 1) / num_threads;
        int group_start = thread_id * groups_per_thread;
        int group_end = (group_start + groups_per_thread > total_groups) ? total_groups : group_start + groups_per_thread;

        for (int g = group_start; g < group_end; g++) {
            for (int k = 0; k < groupSize; k++) {
                int idx0 = g * 2 * groupSize + k;
                int idx1 = idx0 + groupSize;
                int indexWeight = k * total_groups;

                complexFloat temp0 = complexAdd(butterfly[idx0], butterfly[idx1]);
                complexFloat temp1 = complexSub(butterfly[idx0], butterfly[idx1]);

                butterfly[idx0] = complexMul(temp0, getWeight(0, indexWeight, fftLength));   // W0
                butterfly[idx1] = complexMul(temp1, getWeight(1, indexWeight, fftLength));   // W1
            }
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

// Radix-2 FFT DIT
void* fft_radix2_DIT(void *threadarg)
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

    pthread_barrier_wait(barrier); // Wait for malloc

    // Step 1: Initialize butterfly array with bit-reversed input
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

    pthread_barrier_wait(barrier); // Wait for all to finish input copy

    // Step 2: FFT Computation, stage-by-stage
    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = pow(fftRadix, fftStage - 1 - stage);
        int groupSize = pow(fftRadix, stage);

        // Work on different groups per thread
        int groups_per_thread = (groupNum + num_threads - 1) / num_threads;
        int group_start = thread_id * groups_per_thread;
        int group_end = (group_start + groups_per_thread > groupNum) ? groupNum : group_start + groups_per_thread;

        for (int j = group_start; j < group_end; j++) {
            for (int k = 0; k < groupSize; k++) {
                int indexButterfly[fftRadix];
                for (int m = 0; m < fftRadix; m++) {
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                int indexWeight = k * groupNum;

                // Radix-2 butterfly
                complexFloat tmp0 = complexMul(butterfly[indexButterfly[0]], getWeight(0, indexWeight, fftLength));
                complexFloat tmp1 = complexMul(butterfly[indexButterfly[1]], getWeight(1, indexWeight, fftLength));

                butterfly[indexButterfly[0]] = complexAdd(tmp0, tmp1);
                butterfly[indexButterfly[1]] = complexSub(tmp0, tmp1);
            }
        }
        pthread_barrier_wait(barrier); // Synchronize after each stage
    }

    // Step 3: Copy result to output
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


// Radix-4 FFT DIT (Threaded)
void* fft_radix4_DIT(void *threadarg)
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
        int groupNum = pow(fftRadix, fftStage - 1 - stage);
        int groupSize = pow(fftRadix, stage);

        int groups_per_thread = (groupNum + num_threads - 1) / num_threads;
        int group_start = thread_id * groups_per_thread;
        int group_end = (group_start + groups_per_thread > groupNum) ? groupNum : group_start + groups_per_thread;

        for (int j = group_start; j < group_end; j++) {
            for (int k = 0; k < groupSize; k++) {
                int idx[4];
                for (int m = 0; m < 4; m++) {
                    idx[m] = j * groupSize * 4 + k + m * groupSize;
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
                a3 = complexMul(a3, (complexFloat){0, -1});  // Multiply by -j

                butterfly[idx[0]] = complexAdd(a0, a1);
                butterfly[idx[1]] = complexAdd(a2, a3);
                butterfly[idx[2]] = complexSub(a0, a1);
                butterfly[idx[3]] = complexSub(a2, a3);
            }
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

// Radix-4 FFT DIF (Threaded)
void* fft_radix4_DIF(void *threadarg)
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
        if (IFFT) {
            butterfly[i].re = input[i].re;
            butterfly[i].im = -input[i].im;
        } else {
            butterfly[i] = input[i];
        }
    }

    pthread_barrier_wait(barrier);

    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = pow(fftRadix, stage);
        int groupSize = pow(fftRadix, fftStage - 1 - stage);

        int groups_per_thread = (groupNum + num_threads - 1) / num_threads;
        int group_start = thread_id * groups_per_thread;
        int group_end = (group_start + groups_per_thread > groupNum) ? groupNum : group_start + groups_per_thread;

        for (int j = group_start; j < group_end; j++) {
            for (int k = 0; k < groupSize; k++) {
                int idx[4];
                for (int m = 0; m < 4; m++) {
                    idx[m] = j * groupSize * 4 + k + m * groupSize;
                }
                int indexWeight = k * groupNum;

                complexFloat a0 = complexAdd(butterfly[idx[0]], butterfly[idx[2]]);
                complexFloat a1 = complexAdd(butterfly[idx[1]], butterfly[idx[3]]);
                complexFloat a2 = complexSub(butterfly[idx[0]], butterfly[idx[2]]);
                complexFloat a3 = complexSub(butterfly[idx[1]], butterfly[idx[3]]);
                a3 = complexMul(a3, (complexFloat){0, -1});  // Multiply by -j

                complexFloat t0 = complexAdd(a0, a1);
                complexFloat t1 = complexAdd(a2, a3);
                complexFloat t2 = complexSub(a0, a1);
                complexFloat t3 = complexSub(a2, a3);

                butterfly[idx[0]] = complexMul(t0, getWeight(0, indexWeight, fftLength));
                butterfly[idx[1]] = complexMul(t1, getWeight(1, indexWeight, fftLength));
                butterfly[idx[2]] = complexMul(t2, getWeight(2, indexWeight, fftLength));
                butterfly[idx[3]] = complexMul(t3, getWeight(3, indexWeight, fftLength));
            }
        }
        pthread_barrier_wait(barrier);
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


void* fft_radix8_DIF(void *threadarg)
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
        if (IFFT) {
            butterfly[i].re = input[i].re;
            butterfly[i].im = -input[i].im;
        } else {
            butterfly[i] = input[i];
        }
    }
    pthread_barrier_wait(barrier);

    for (int stage = 0; stage < fftStage; stage++) {
        int groupNum = pow(fftRadix, stage);
        int groupSize = pow(fftRadix, fftStage - 1 - stage);

        int groups_per_thread = (groupNum + num_threads - 1) / num_threads;
        int group_start = thread_id * groups_per_thread;
        int group_end = (group_start + groups_per_thread > groupNum) ? groupNum : group_start + groups_per_thread;

        for (int j = group_start; j < group_end; j++) {
            for (int k = 0; k < groupSize; k++) {
                int idx[8];
                for (int m = 0; m < 8; m++) {
                    idx[m] = j * groupSize * 8 + k + m * groupSize;
                }
                int indexWeight = k * groupNum;

                complexFloat a0 = complexAdd(butterfly[idx[0]], butterfly[idx[4]]);
                complexFloat a1 = complexAdd(butterfly[idx[2]], butterfly[idx[6]]);
                complexFloat a2 = complexAdd(butterfly[idx[1]], butterfly[idx[5]]);
                complexFloat a3 = complexAdd(butterfly[idx[3]], butterfly[idx[7]]);
                complexFloat a4 = complexSub(butterfly[idx[0]], butterfly[idx[4]]);
                complexFloat a5 = complexSub(butterfly[idx[2]], butterfly[idx[6]]);
                complexFloat a6 = complexSub(butterfly[idx[1]], butterfly[idx[5]]);
                complexFloat a7 = complexSub(butterfly[idx[3]], butterfly[idx[7]]);

                a5 = complexMul(a5, (complexFloat){0, -1});
                a7 = complexMul(a7, (complexFloat){0, -1});

                complexFloat b0 = complexAdd(a0, a1);
                complexFloat b1 = complexAdd(a2, a3);
                complexFloat b2 = complexSub(a0, a1);
                complexFloat b3 = complexSub(a2, a3);
                complexFloat b4 = complexAdd(a4, a5);
                complexFloat b5 = complexAdd(a6, a7);
                complexFloat b6 = complexSub(a4, a5);
                complexFloat b7 = complexSub(a6, a7);

                b5 = complexMul(b5, (complexFloat){cos(M_PI/4), -sin(M_PI/4)});
                b3 = complexMul(b3, (complexFloat){0, -1});
                b7 = complexMul(b7, (complexFloat){cos(3*M_PI/4), -sin(3*M_PI/4)});

                butterfly[idx[0]] = complexMul(complexAdd(b0, b1), getWeight(0, indexWeight, fftLength));
                butterfly[idx[1]] = complexMul(complexAdd(b4, b5), getWeight(1, indexWeight, fftLength));
                butterfly[idx[2]] = complexMul(complexAdd(b2, b3), getWeight(2, indexWeight, fftLength));
                butterfly[idx[3]] = complexMul(complexAdd(b6, b7), getWeight(3, indexWeight, fftLength));
                butterfly[idx[4]] = complexMul(complexSub(b0, b1), getWeight(4, indexWeight, fftLength));
                butterfly[idx[5]] = complexMul(complexSub(b4, b5), getWeight(5, indexWeight, fftLength));
                butterfly[idx[6]] = complexMul(complexSub(b2, b3), getWeight(6, indexWeight, fftLength));
                butterfly[idx[7]] = complexMul(complexSub(b6, b7), getWeight(7, indexWeight, fftLength));
            }
        }
        pthread_barrier_wait(barrier);
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

void* fft_radix8_DIT(void *threadarg)
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

        int groups_per_thread = (groupNum + num_threads - 1) / num_threads;
        int group_start = thread_id * groups_per_thread;
        int group_end = (group_start + groups_per_thread > groupNum) ? groupNum : group_start + groups_per_thread;

        for (int j = group_start; j < group_end; j++) {
            for (int k = 0; k < groupSize; k++) {
                int idx[8];
                for (int m = 0; m < 8; m++) {
                    idx[m] = j * groupSize * 8 + k + m * groupSize;
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