#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "fft.h"

//#define fftRadix 4     // ← switch from 8 to 4
//#define fftStage 8     // ← for a 64-point FFT, since 4³ = 64
//#define fftLength pow(fftRadix, fftStage) 



int main(void){
    int fftRadix = 2;     //
    int fftStage = 24;    
    int fftLength = pow(fftRadix, fftStage); 
    FILE *fpsignal = fopen("0signal.txt","r");
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

    fft(fftInput, spectrum, false, fftRadix, fftStage, fftLength);
    for(int i = 0; i < fftLength; i++){
        fprintf(fpspectrum, "%.5f \t%.5f\n", spectrum[i].re, spectrum[i].im);
    }
    fclose(fpspectrum);
    printf("write fft output spectrum finished !\n");

    fft(spectrum, ifftOutput, true, fftRadix, fftStage, fftLength);
    for(int i = 0; i < fftLength; i++){
        fprintf(fpifftOutput, "%f\t%f\n", ifftOutput[i].re, ifftOutput[i].im);
    }
    fclose(fpifftOutput);
    printf("write ifft output data finished !\n");

    clock_t start_fft = clock();
    fft(fftInput, spectrum, false, fftRadix, fftStage, fftLength);
    clock_t end_fft = clock();
    double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
    printf("FFT time-Radix2: %f seconds\n", time_fft);

    fftRadix = 4;    
    fftStage = 12;     
    fftLength = pow(fftRadix, fftStage); 

     start_fft = clock();
    fft(fftInput, spectrum, false, fftRadix, fftStage, fftLength);
     end_fft = clock();
     time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
    printf("FFT time-Radix4: %f seconds\n", time_fft);

    fftRadix = 8;     
    fftStage = 8;    
    fftLength = pow(fftRadix, fftStage); 

     start_fft = clock();
    fft(fftInput, spectrum, false, fftRadix, fftStage, fftLength);
     end_fft = clock();
     time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
    printf("FFT time-Radix8: %f seconds\n", time_fft);



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
