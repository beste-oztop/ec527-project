/* perf stat -e cache-references,cache-misses 
./fft_main 2 12 1 
gcc -fopenmp -O1 -o fft_main main.c -lm
OMP_NUM_THREADS=8 ./fft_main fft_input.txt 2 12 1
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

typedef unsigned int u32;
typedef unsigned short u16;
typedef char u8;
#define DIT 0 
#define DIF 1 
#define ARRAY_SIZE 1024*2
typedef struct _complexFloat{
    float re;
    float im;
}complexFloat;


int fft_radix2_DIF( complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix2_DIT(complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix4_DIT( complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix4_DIF( complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix8_DIT(complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int fft_radix8_DIF( complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType);
int FFT2D_Radix2_DIT(complexFloat *data, int height, int width);
int FFT2D_Radix2_DIF(complexFloat *data, int height, int width);
int FFT2D_Radix4_DIT(complexFloat *data, int height, int width);
int FFT2D_Radix4_DIF(complexFloat *data, int height, int width);
int FFT2D_Radix8_DIT(complexFloat *data, int height, int width);
int FFT2D_Radix8_DIF(complexFloat *data, int height, int width);
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
    int THREADS = 8; // Default number of threads
    char *env_ONT = getenv("OMP_NUM_THREADS");
    if (argc != 5) {
        printf("Usage: %s <input_file.txt> <fftRadix: 2|4|8> <fftStage> <fftType: 0 (DIT) | 1 (DIF)>\n", argv[0]);
        return 1;
    }

    #pragma omp parallel for
    for(i=0; i<1; i++) {
        ognt = omp_get_num_threads();
    }

    printf("omp's default number of threads is %d\n", ognt);

    /* If this is illegal (0 or less), default to the "#define THREADS"
        value that is defined above */
    if (ognt <= 0) {
        if (THREADS != ognt) {
        printf("Overriding with #define THREADS value %d\n", THREADS);
        ognt = THREADS;
        }
    }

    omp_set_num_threads(ognt);
    final_answer = wakeup_delay();
    /* Once again ask OpenMP how many threads it is going to use */
    #pragma omp parallel for
    for(i=0; i<1; i++) {
        ognt = omp_get_num_threads();
    }
    printf("Using %d threads for OpenMP\n", ognt);
    


    // Read the matrix input data
    complexFloat *data;
    int height, width;
    if (!readMatrix(argv[1], &data, &height, &width)) {
        return 1;
    }
    int fftRadix = atoi(argv[2]);
    int fftStage = atoi(argv[3]);
    int fftType = atoi(argv[4]);
    int fftLength = (int)pow(fftRadix, fftStage);
    printf("fftLength = %d\n", fftLength);
    printf("fftType = %d\n", fftType);

    FILE *fpsignal = fopen("fft_input.txt","r");
    FILE *fpfftInput = fopen("1fftInput.txt","w");
    FILE *fpspectrum = fopen("2spectrum.txt","w");
    FILE *fpifftOutput = fopen("3ifftOutput.txt","w");
    if (fpsignal == NULL ||
        fpfftInput == NULL ||
        fpspectrum == NULL ||
        fpifftOutput == NULL){
        printf("Error: open file failed!");
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

    // Switch case between different FFT types
    switch (fftRadix){
        case 2:
            if(fftType == DIT){
                clock_gettime(CLOCK_REALTIME, &time_start);
                fft_radix2_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix2 DIT: %f seconds\n", time_fft);

                // 2DFFT Radix2 DIT
                clock_gettime(CLOCK_REALTIME, &time_start);
                FFT2D_Radix2_DIT(data, height, width);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_2fft = interval(time_start, time_stop);
                FILE *outputFile = fopen("2DFFT_Radix2_output_c_DIT.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < height * width; i++) {
                    fprintf(outputFile, "%f %f\n", data[i].re, data[i].im);
                }
                fclose(outputFile);
                free(data);
                printf("2DFFT time-Radix2 DIT: %f seconds\n", time_2fft);

            }else{
                clock_gettime(CLOCK_REALTIME, &time_start);
                fft_radix2_DIF(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix2 DIF: %f seconds\n", time_fft);

                // 2DFFT Radix2 DIT
                clock_gettime(CLOCK_REALTIME, &time_start);
                FFT2D_Radix2_DIF(data, height, width);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_2fft = interval(time_start, time_stop);
                FILE *outputFile = fopen("2DFFT_Radix2_output_c_DIF.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < height * width; i++) {
                    fprintf(outputFile, "%f %f\n", data[i].re, data[i].im);
                }
                fclose(outputFile);
                free(data);
                printf("2DFFT time-Radix2 DIF: %f seconds\n", time_2fft);
            }
            break;
        case 4:
            if(fftType == DIT){
                clock_gettime(CLOCK_REALTIME, &time_start);
                fft_radix4_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix4 DIT: %f seconds\n", time_fft);

                // 2DFFT Radix2 DIT
                clock_gettime(CLOCK_REALTIME, &time_start);
                FFT2D_Radix4_DIT(data, height, width);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_2fft = interval(time_start, time_stop);
                FILE *outputFile = fopen("2DFFT_Radix4_output_c_DIT.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < height * width; i++) {
                    fprintf(outputFile, "%f %f\n", data[i].re, data[i].im);
                }
                fclose(outputFile);
                free(data);
                printf("2DFFT time-Radix4 DIT: %f seconds\n", time_2fft);
            }else{
                clock_gettime(CLOCK_REALTIME, &time_start);
                fft_radix4_DIF(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix4 DIF: %f seconds\n", time_fft);

                // 2DFFT Radix2 DIT
                clock_gettime(CLOCK_REALTIME, &time_start);
                FFT2D_Radix4_DIF(data, height, width);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_2fft = interval(time_start, time_stop);
                FILE *outputFile = fopen("2DFFT_Radix4_output_c_DIF.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < height * width; i++) {
                    fprintf(outputFile, "%f %f\n", data[i].re, data[i].im);
                }
                fclose(outputFile);
                free(data);
                printf("2DFFT time-Radix4 DIF: %f seconds\n", time_2fft);

            }
            break;
        case 8:
            if(fftType == DIT){
                clock_gettime(CLOCK_REALTIME, &time_start);
                fft_radix8_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix8 DIT: %f seconds\n", time_fft);

                // 2DFFT Radix2 DIT
                clock_gettime(CLOCK_REALTIME, &time_start);
                FFT2D_Radix8_DIT(data, height, width);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_2fft = interval(time_start, time_stop);
                FILE *outputFile = fopen("2DFFT_Radix8_output_c_DIT.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < height * width; i++) {
                    fprintf(outputFile, "%f %f\n", data[i].re, data[i].im);
                }
                fclose(outputFile);
                free(data);
                printf("2DFFT time-Radix8 DIT: %f seconds\n", time_2fft);
            }else{
                clock_gettime(CLOCK_REALTIME, &time_start);
                fft_radix8_DIF(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_fft = interval(time_start, time_stop);
                printf("FFT time-Radix8 DIF: %f seconds\n", time_fft);

                // 2DFFT Radix2 DIT
                clock_gettime(CLOCK_REALTIME, &time_start);
                FFT2D_Radix8_DIF(data, height, width);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                double time_2fft = interval(time_start, time_stop);
                FILE *outputFile = fopen("2DFFT_Radix8_output_c_DIF.txt", "w");
                if (outputFile == NULL) {
                    printf("Error: Could not open file for writing.\n");
                    return 1;
                }
                for (int i = 0; i < height * width; i++) {
                    fprintf(outputFile, "%f %f\n", data[i].re, data[i].im);
                }
                fclose(outputFile);
                free(data);
                printf("2DFFT time-Radix8 DIF: %f seconds\n", time_2fft);
            }
            break;
    }

    free(signal);
    free(fftInput);
    free(spectrum);
    free(ifftOutput);
    signal = NULL;
    fftInput = NULL;
    spectrum = NULL;
    ifftOutput = NULL;
    return 0;
}


// Radix-2 FFT DIF
int fft_radix2_DIF( complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType ){
    complexFloat *butterfly = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  i;
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
        groupNum =  pow(fftRadix, i);
        groupSize =  pow(fftRadix, fftStage-1 - i);
        for(int j = 0; j < groupNum; j++){
            for(int k = 0; k < groupSize; k++){
                for(int m = 0; m < fftRadix; m++){
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;
                tmpButterfly[0][0] = complexAdd(butterfly[indexButterfly[0]], butterfly[indexButterfly[1]]);
                tmpButterfly[0][1] = complexSub(butterfly[indexButterfly[0]], butterfly[indexButterfly[1]]);
                butterfly[indexButterfly[0]] = complexMul(tmpButterfly[0][0], getWeight(0,indexWeight, fftLength));
                butterfly[indexButterfly[1]] = complexMul(tmpButterfly[0][1], getWeight(1,indexWeight, fftLength));
                    
            }
        }
    }

    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder = reverseBit(i,fftRadix,fftLength) ;
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

// Radix-4 FFT DIF
int fft_radix4_DIF( complexFloat *input,  complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType ){
    complexFloat *butterfly = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder = i;
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
        groupNum = pow(fftRadix, i);
        groupSize = pow(fftRadix, fftStage-1 - i);
        for(int j = 0; j < groupNum; j++){
            for(int k = 0; k < groupSize; k++){
                for(int m = 0; m < fftRadix; m++){
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;
                tmpButterfly[0][0] = complexAdd(butterfly[indexButterfly[0]], butterfly[indexButterfly[2]]);
                tmpButterfly[0][2] = complexSub(butterfly[indexButterfly[0]], butterfly[indexButterfly[2]]);
                tmpButterfly[0][1] = complexAdd(butterfly[indexButterfly[1]], butterfly[indexButterfly[3]]);
                tmpButterfly[0][3] = complexSub(butterfly[indexButterfly[1]], butterfly[indexButterfly[3]]);
                tmpButterfly[0][3] = complexMul(tmpButterfly[0][3], (complexFloat){0,-1});
                tmpButterfly[1][0] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][1]);
                tmpButterfly[1][1] = complexAdd(tmpButterfly[0][2], tmpButterfly[0][3]);
                tmpButterfly[1][2] = complexSub(tmpButterfly[0][0], tmpButterfly[0][1]);
                tmpButterfly[1][3] = complexSub(tmpButterfly[0][2], tmpButterfly[0][3]);
                butterfly[indexButterfly[0]] = complexMul(tmpButterfly[1][0], getWeight(0,indexWeight, fftLength));
                butterfly[indexButterfly[1]] = complexMul(tmpButterfly[1][1], getWeight(1,indexWeight, fftLength));
                butterfly[indexButterfly[2]] = complexMul(tmpButterfly[1][2], getWeight(2,indexWeight, fftLength));
                butterfly[indexButterfly[3]] = complexMul(tmpButterfly[1][3], getWeight(3,indexWeight, fftLength));
                    
            }
        }
    }

    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder = reverseBit(i,fftRadix,fftLength) ;
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


// Radix-8 FFT DIF
int fft_radix8_DIF(complexFloat *input, complexFloat *output, bool IFFT, int fftRadix, int fftStage, int fftLength, int fftType ){
    complexFloat *butterfly = (complexFloat*) malloc(sizeof(complexFloat) * fftLength);
    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder = i;
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
        groupNum = pow(fftRadix, i);
        groupSize =  pow(fftRadix, fftStage-1 - i);
        for(int j = 0; j < groupNum; j++){
            for(int k = 0; k < groupSize; k++){
                for(int m = 0; m < fftRadix; m++){
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;
                tmpButterfly[0][0] = complexAdd(butterfly[indexButterfly[0]], butterfly[indexButterfly[4]]);
                tmpButterfly[0][4] = complexSub(butterfly[indexButterfly[0]], butterfly[indexButterfly[4]]);
                tmpButterfly[0][2] = complexAdd(butterfly[indexButterfly[2]], butterfly[indexButterfly[6]]);
                tmpButterfly[0][6] = complexSub(butterfly[indexButterfly[2]], butterfly[indexButterfly[6]]);
                tmpButterfly[0][1] = complexAdd(butterfly[indexButterfly[1]], butterfly[indexButterfly[5]]);
                tmpButterfly[0][5] = complexSub(butterfly[indexButterfly[1]], butterfly[indexButterfly[5]]);
                tmpButterfly[0][3] = complexAdd(butterfly[indexButterfly[3]], butterfly[indexButterfly[7]]);
                tmpButterfly[0][7] = complexSub(butterfly[indexButterfly[3]], butterfly[indexButterfly[7]]);
                tmpButterfly[0][6] = complexMul(tmpButterfly[0][6], (complexFloat){0,-1});
                tmpButterfly[0][7] = complexMul(tmpButterfly[0][7], (complexFloat){0,-1});
                tmpButterfly[1][0] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][2]);
                tmpButterfly[1][2] = complexSub(tmpButterfly[0][0], tmpButterfly[0][2]);
                tmpButterfly[1][4] = complexAdd(tmpButterfly[0][4], tmpButterfly[0][6]);
                tmpButterfly[1][6] = complexSub(tmpButterfly[0][4], tmpButterfly[0][6]);
                tmpButterfly[1][1] = complexAdd(tmpButterfly[0][1], tmpButterfly[0][3]);
                tmpButterfly[1][3] = complexSub(tmpButterfly[0][1], tmpButterfly[0][3]);
                tmpButterfly[1][5] = complexAdd(tmpButterfly[0][5], tmpButterfly[0][7]);
                tmpButterfly[1][7] = complexSub(tmpButterfly[0][5], tmpButterfly[0][7]);
                tmpButterfly[1][5] = complexMul(tmpButterfly[1][5], (complexFloat){cos(M_PI/4),-sin(M_PI/4)});
                tmpButterfly[1][3] = complexMul(tmpButterfly[1][3], (complexFloat){0,-1});
                tmpButterfly[1][7] = complexMul(tmpButterfly[1][7], (complexFloat){cos(3*M_PI/4),-sin(3*M_PI/4)});
                tmpButterfly[2][0] = complexAdd(tmpButterfly[1][0], tmpButterfly[1][1]);
                tmpButterfly[2][1] = complexAdd(tmpButterfly[1][4], tmpButterfly[1][5]);
                tmpButterfly[2][2] = complexAdd(tmpButterfly[1][2], tmpButterfly[1][3]);
                tmpButterfly[2][3] = complexAdd(tmpButterfly[1][6], tmpButterfly[1][7]);
                tmpButterfly[2][4] = complexSub(tmpButterfly[1][0], tmpButterfly[1][1]);
                tmpButterfly[2][5] = complexSub(tmpButterfly[1][4], tmpButterfly[1][5]);
                tmpButterfly[2][6] = complexSub(tmpButterfly[1][2], tmpButterfly[1][3]);
                tmpButterfly[2][7] = complexSub(tmpButterfly[1][6], tmpButterfly[1][7]);
                butterfly[indexButterfly[0]] = complexMul(tmpButterfly[2][0], getWeight(0,indexWeight, fftLength));
                butterfly[indexButterfly[1]] = complexMul(tmpButterfly[2][1], getWeight(1,indexWeight, fftLength));
                butterfly[indexButterfly[2]] = complexMul(tmpButterfly[2][2], getWeight(2,indexWeight, fftLength));
                butterfly[indexButterfly[3]] = complexMul(tmpButterfly[2][3], getWeight(3,indexWeight, fftLength));
                butterfly[indexButterfly[4]] = complexMul(tmpButterfly[2][4], getWeight(4,indexWeight, fftLength));
                butterfly[indexButterfly[5]] = complexMul(tmpButterfly[2][5], getWeight(5,indexWeight, fftLength));
                butterfly[indexButterfly[6]] = complexMul(tmpButterfly[2][6], getWeight(6,indexWeight, fftLength));
                butterfly[indexButterfly[7]] = complexMul(tmpButterfly[2][7], getWeight(7,indexWeight, fftLength));
            }
        }
    }

    for(u32 i = 0; i < fftLength; i++){
        u32 iOrder;
        iOrder =  reverseBit(i,fftRadix,fftLength) ;
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


int readMatrix(const char *filename, complexFloat **data, int *height, int *width) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return 0;
    }

    char line[ARRAY_SIZE];
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
    *data = malloc(h * w * sizeof(complexFloat));
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
            double val = atof(token);
            (*data)[idx].re = val; // Assign real part
            (*data)[idx].im = 0.0; // Assign imaginary part as 0
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


#include <alloca.h>  // for alloca()

int FFT2D_Radix2_DIT(complexFloat *data, int height, int width) {
    // ----------- Row-wise FFT ------------
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        complexFloat *row = alloca(width * sizeof(complexFloat));

        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft_radix2_DIT(row, row, false, 2, (int)log2(width), width, 0);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }

    // ----------- Column-wise FFT ------------
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        complexFloat *col = alloca(height * sizeof(complexFloat));

        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft_radix2_DIT(col, col, false, 2, (int)log2(height), height, 0);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    return 0;
}

int FFT2D_Radix2_DIF(complexFloat *data, int height, int width) {
     // ----------- Row-wise FFT ------------
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        complexFloat *row = alloca(width * sizeof(complexFloat));

        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft_radix2_DIF(row, row, false, 2, (int)log2(width), width, 1);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }

    // ----------- Column-wise FFT ------------
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        complexFloat *col = alloca(height * sizeof(complexFloat));

        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft_radix2_DIF(col, col, false, 2, (int)log2(height), height, 1);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    return 0;
}

int FFT2D_Radix4_DIT(complexFloat *data, int height, int width){
     // ----------- Row-wise FFT ------------
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        complexFloat *row = alloca(width * sizeof(complexFloat));

        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft_radix4_DIT(row, row, false, 4, (int)log2(width)/2, width, 0);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }

    // ----------- Column-wise FFT ------------
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        complexFloat *col = alloca(height * sizeof(complexFloat));

        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft_radix4_DIT(col, col, false, 4, (int)log2(height)/2, height, 0);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    return 0;
} 

int FFT2D_Radix4_DIF(complexFloat *data, int height, int width){
      // ----------- Row-wise FFT ------------
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        complexFloat *row = alloca(width * sizeof(complexFloat));

        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft_radix4_DIF(row, row, false, 4, (int)log2(width)/2, width, 1);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }

    // ----------- Column-wise FFT ------------
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        complexFloat *col = alloca(height * sizeof(complexFloat));

        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft_radix4_DIF(col, col, false, 4, (int)log2(height)/2, height, 1);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    return 0;
}

int FFT2D_Radix8_DIT(complexFloat *data, int height, int width) {
       // ----------- Row-wise FFT ------------
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        complexFloat *row = alloca(width * sizeof(complexFloat));

        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft_radix8_DIT(row, row, false, 8, (int)log2(width)/3, width, 0);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }

    // ----------- Column-wise FFT ------------
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        complexFloat *col = alloca(height * sizeof(complexFloat));

        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft_radix8_DIT(col, col, false, 8, (int)log2(height)/3, height, 0);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    return 0;
}

int FFT2D_Radix8_DIF(complexFloat *data, int height, int width) {
      // ----------- Row-wise FFT ------------
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        complexFloat *row = alloca(width * sizeof(complexFloat));

        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft_radix8_DIF(row, row, false, 8, (int)log2(width)/3, width, 1);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }

    // ----------- Column-wise FFT ------------
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        complexFloat *col = alloca(height * sizeof(complexFloat));

        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft_radix8_DIF(col, col, false, 8, (int)log2(height)/3, height, 1);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    return 0;
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