/* perf stat -e cache-references,cache-misses ./fft_main 2 12 1 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>

typedef unsigned int u32;
typedef unsigned short u16;
typedef char u8;
#define DIT 0 
#define DIF 1 
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
static complexFloat complexAdd( complexFloat A, complexFloat B);
static complexFloat complexSub( complexFloat A, complexFloat B);
static complexFloat complexMul( complexFloat A, complexFloat B);
static u32 reverseBit( u32 n, int fftRadix, int fftLength);
static complexFloat getWeight( int iButterfly, int indexWeight, int fftLength);

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <fftRadix: 2|4|8> <fftStage> <fftType: 0 (DIT) | 1 (DIF)>\n", argv[0]);
        return 1;
    }

    int fftRadix = atoi(argv[1]);
    int fftStage = atoi(argv[2]);
    int fftType = atoi(argv[3]);
    int fftLength = (int)pow(fftRadix, fftStage);
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

    for (int i = 0; i < fftLength; i++){
        fscanf(fpsignal,"%f", &signal[i]);
    }
    fclose(fpsignal);

    for (int i = 0; i < fftLength; i++){
        fftInput[i].re = signal[i];
        fftInput[i].im = 0;
        fprintf(fpfftInput, "%f\t%f\n", fftInput[i].re, fftInput[i].im);
    }
    fclose(fpfftInput);
    printf("read input signal data finished !\n");

    // Replace PAPI code with perf_event_open counters
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.size = sizeof(struct perf_event_attr);
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    int fd_cache_refs = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd_cache_refs == -1) {
        perror("perf_event_open (cache references) failed");
        return 1;
    }
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.size = sizeof(struct perf_event_attr);
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CACHE_MISSES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    int fd_cache_misses = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd_cache_misses == -1) {
        perror("perf_event_open (cache misses) failed");
        return 1;
    }

    switch (fftRadix){
        case 2:
            // Start performance counters
            ioctl(fd_cache_refs, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_cache_refs, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_ENABLE, 0);

            if(fftType == DIT) {
                clock_t start_fft = clock();
                fft_radix2_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_t end_fft = clock();
                double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
                printf("FFT time-Radix2 DIT: %f seconds\n", time_fft);
            } else {
                clock_t start_fft = clock();
                fft_radix2_DIF(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 1);
                clock_t end_fft = clock();
                double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
                printf("FFT time-Radix2 DIF: %f seconds\n", time_fft);
            }
            // Stop performance counters
            ioctl(fd_cache_refs, PERF_EVENT_IOC_DISABLE, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_DISABLE, 0);

            {
                long long cache_refs = 0, cache_misses = 0;
                read(fd_cache_refs, &cache_refs, sizeof(cache_refs));
                read(fd_cache_misses, &cache_misses, sizeof(cache_misses));
                printf("Cache references: %lld\n", cache_refs);
                printf("Cache misses: %lld\n", cache_misses);
            }
            break;

        case 4:
            // Similar changes for Radix 4
            ioctl(fd_cache_refs, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_cache_refs, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_ENABLE, 0);

            if(fftType == DIT) {
                clock_t start_fft = clock();
                fft_radix4_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_t end_fft = clock();
                double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
                printf("FFT time-Radix4 DIT: %f seconds\n", time_fft);
            } else {
                clock_t start_fft = clock();
                fft_radix4_DIF(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 1);
                clock_t end_fft = clock();
                double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
                printf("FFT time-Radix4 DIF: %f seconds\n", time_fft);
            }
            ioctl(fd_cache_refs, PERF_EVENT_IOC_DISABLE, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_DISABLE, 0);

            {
                long long cache_refs = 0, cache_misses = 0;
                read(fd_cache_refs, &cache_refs, sizeof(cache_refs));
                read(fd_cache_misses, &cache_misses, sizeof(cache_misses));
                printf("Cache references: %lld\n", cache_refs);
                printf("Cache misses: %lld\n", cache_misses);
            }
            break;

        case 8:
            // Similar changes for Radix 8
            ioctl(fd_cache_refs, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_cache_refs, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_ENABLE, 0);

            if(fftType == DIT) {
                clock_t start_fft = clock();
                fft_radix8_DIT(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 0);
                clock_t end_fft = clock();
                double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
                printf("FFT time-Radix8 DIT: %f seconds\n", time_fft);
            } else {
                clock_t start_fft = clock();
                fft_radix8_DIF(fftInput, spectrum, false, fftRadix, fftStage, fftLength, 1);
                clock_t end_fft = clock();
                double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
                printf("FFT time-Radix8 DIF: %f seconds\n", time_fft);
            }
            ioctl(fd_cache_refs, PERF_EVENT_IOC_DISABLE, 0);
            ioctl(fd_cache_misses, PERF_EVENT_IOC_DISABLE, 0);

            {
                long long cache_refs = 0, cache_misses = 0;
                read(fd_cache_refs, &cache_refs, sizeof(cache_refs));
                read(fd_cache_misses, &cache_misses, sizeof(cache_misses));
                printf("Cache references: %lld\n", cache_refs);
                printf("Cache misses: %lld\n", cache_misses);
            }
            break;
    }

    close(fd_cache_refs);
    close(fd_cache_misses);

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