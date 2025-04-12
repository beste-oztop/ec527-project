#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define DIT 0
#define DIF 1

typedef unsigned int u32;
typedef struct {
    float re;
    float im;
} complexFloat;

int fftLength;
int fftRadix;
int fftStage;
int fftType;

static complexFloat complexAdd(complexFloat A, complexFloat B) {
    complexFloat result;
    result.re = A.re + B.re;
    result.im = A.im + B.im;
    return result;
}

static complexFloat complexSub(complexFloat A, complexFloat B) {
    complexFloat result;
    result.re = A.re - B.re;
    result.im = A.im - B.im;
    return result;
}

static complexFloat complexMul(complexFloat A, complexFloat B) {
    complexFloat result;
    result.re = A.re * B.re - A.im * B.im;
    result.im = A.im * B.re + A.re * B.im;
    return result;
}

static complexFloat getWeight(int iButterfly, int indexWeight) {
    complexFloat result;
    result.re = cos(2 * M_PI * iButterfly * indexWeight / fftLength);
    result.im = -sin(2 * M_PI * iButterfly * indexWeight / fftLength);
    return result;
}

static u32 reverseBit(u32 n) {
    u32 bitWidth = 0;
    while ((1 << bitWidth) < fftLength) bitWidth++;
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1);
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2);
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4);
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8);
    n = ((n & 0xFFFF0000) >> 16) | ((n & 0x0000FFFF) << 16);
    return n >> (32 - bitWidth);
}

int fft(complexFloat *input, complexFloat *output, bool IFFT) {
    complexFloat *butterfly = malloc(sizeof(complexFloat) * fftLength);
    for (u32 i = 0; i < fftLength; i++) {
        u32 iOrder = (fftType == DIT) ? reverseBit(i) : i;
        if (IFFT) {
            butterfly[iOrder].re = input[i].re;
            butterfly[iOrder].im = -input[i].im;
        } else {
            butterfly[iOrder] = input[i];
        }
    }

    complexFloat tmpButterfly[3][8];
    int indexButterfly[8];
    int indexWeight;

    for (int i = 0; i < fftStage; i++) {
        int groupNum = (fftType == DIT) ? pow(fftRadix, fftStage - 1 - i) : pow(fftRadix, i);
        int groupSize = (fftType == DIT) ? pow(fftRadix, i) : pow(fftRadix, fftStage - 1 - i);

        for (int j = 0; j < groupNum; j++) {
            for (int k = 0; k < groupSize; k++) {
                for (int m = 0; m < fftRadix; m++) {
                    indexButterfly[m] = j * groupSize * fftRadix + k + m * groupSize;
                }
                indexWeight = k * groupNum;

                for (int m = 0; m < fftRadix; m++) {
                    tmpButterfly[0][m] = complexMul(butterfly[indexButterfly[m]], getWeight(m, indexWeight));
                }

                if (fftRadix == 2) {
                    butterfly[indexButterfly[0]] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][1]);
                    butterfly[indexButterfly[1]] = complexSub(tmpButterfly[0][0], tmpButterfly[0][1]);
                } else if (fftRadix == 4) {
                    tmpButterfly[1][0] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][2]);
                    tmpButterfly[1][2] = complexSub(tmpButterfly[0][0], tmpButterfly[0][2]);
                    tmpButterfly[1][1] = complexAdd(tmpButterfly[0][1], tmpButterfly[0][3]);
                    tmpButterfly[1][3] = complexSub(tmpButterfly[0][1], tmpButterfly[0][3]);
                    tmpButterfly[1][3] = complexMul(tmpButterfly[1][3], (complexFloat){0, -1});
                    butterfly[indexButterfly[0]] = complexAdd(tmpButterfly[1][0], tmpButterfly[1][1]);
                    butterfly[indexButterfly[1]] = complexAdd(tmpButterfly[1][2], tmpButterfly[1][3]);
                    butterfly[indexButterfly[2]] = complexSub(tmpButterfly[1][0], tmpButterfly[1][1]);
                    butterfly[indexButterfly[3]] = complexSub(tmpButterfly[1][2], tmpButterfly[1][3]);
                } else if (fftRadix == 8) {
                    complexFloat j1 = {0, -1};
                    complexFloat j2 = {cos(M_PI / 4), -sin(M_PI / 4)};
                    complexFloat j3 = {cos(3 * M_PI / 4), -sin(3 * M_PI / 4)};

                    tmpButterfly[1][0] = complexAdd(tmpButterfly[0][0], tmpButterfly[0][4]);
                    tmpButterfly[1][4] = complexSub(tmpButterfly[0][0], tmpButterfly[0][4]);
                    tmpButterfly[1][2] = complexAdd(tmpButterfly[0][2], tmpButterfly[0][6]);
                    tmpButterfly[1][6] = complexSub(tmpButterfly[0][2], tmpButterfly[0][6]);
                    tmpButterfly[1][1] = complexAdd(tmpButterfly[0][1], tmpButterfly[0][5]);
                    tmpButterfly[1][5] = complexSub(tmpButterfly[0][1], tmpButterfly[0][5]);
                    tmpButterfly[1][3] = complexAdd(tmpButterfly[0][3], tmpButterfly[0][7]);
                    tmpButterfly[1][7] = complexSub(tmpButterfly[0][3], tmpButterfly[0][7]);
                    tmpButterfly[1][6] = complexMul(tmpButterfly[1][6], j1);
                    tmpButterfly[1][7] = complexMul(tmpButterfly[1][7], j1);

                    tmpButterfly[2][0] = complexAdd(tmpButterfly[1][0], tmpButterfly[1][2]);
                    tmpButterfly[2][2] = complexSub(tmpButterfly[1][0], tmpButterfly[1][2]);
                    tmpButterfly[2][4] = complexAdd(tmpButterfly[1][4], tmpButterfly[1][6]);
                    tmpButterfly[2][6] = complexSub(tmpButterfly[1][4], tmpButterfly[1][6]);
                    tmpButterfly[2][1] = complexAdd(tmpButterfly[1][1], tmpButterfly[1][3]);
                    tmpButterfly[2][3] = complexSub(tmpButterfly[1][1], tmpButterfly[1][3]);
                    tmpButterfly[2][5] = complexAdd(tmpButterfly[1][5], tmpButterfly[1][7]);
                    tmpButterfly[2][7] = complexSub(tmpButterfly[1][5], tmpButterfly[1][7]);

                    tmpButterfly[2][5] = complexMul(tmpButterfly[2][5], j2);
                    tmpButterfly[2][3] = complexMul(tmpButterfly[2][3], j1);
                    tmpButterfly[2][7] = complexMul(tmpButterfly[2][7], j3);

                    butterfly[indexButterfly[0]] = complexAdd(tmpButterfly[2][0], tmpButterfly[2][1]);
                    butterfly[indexButterfly[1]] = complexAdd(tmpButterfly[2][4], tmpButterfly[2][5]);
                    butterfly[indexButterfly[2]] = complexAdd(tmpButterfly[2][2], tmpButterfly[2][3]);
                    butterfly[indexButterfly[3]] = complexAdd(tmpButterfly[2][6], tmpButterfly[2][7]);
                    butterfly[indexButterfly[4]] = complexSub(tmpButterfly[2][0], tmpButterfly[2][1]);
                    butterfly[indexButterfly[5]] = complexSub(tmpButterfly[2][4], tmpButterfly[2][5]);
                    butterfly[indexButterfly[6]] = complexSub(tmpButterfly[2][2], tmpButterfly[2][3]);
                    butterfly[indexButterfly[7]] = complexSub(tmpButterfly[2][6], tmpButterfly[2][7]);
                } else {
                    printf("Error: only radix 2, 4, 8 supported in this benchmark.\n");
                    free(butterfly);
                    return 1;
                }
            }
        }
    }

    for (u32 i = 0; i < fftLength; i++) {
        u32 iOrder = (fftType == DIF) ? reverseBit(i) : i;
        if (IFFT) {
            output[iOrder].re = butterfly[i].re / fftLength;
            output[iOrder].im = -butterfly[i].im / fftLength;
        } else {
            output[iOrder] = butterfly[i];
        }
    }

    free(butterfly);
    return 0;
}

int main() {
    int Ns[] = {64, 256, 1024, 2048, 4096};
    int radixOptions[] = {2, 4, 8};
    const char *radixNames[] = {"Radix-2", "Radix-4", "Radix-8"};

    for (int r = 0; r < 3; ++r) {
        fftRadix = radixOptions[r];
       printf("\n=== Testing %s FFT ===\n", radixNames[r]);
        printf("Radix = %d\n", fftRadix);

        for (int i = 0; i < 5; ++i) {
            fftLength = Ns[i];
            double logValue = log(fftLength) / log(fftRadix);
            if (fabs(logValue - round(logValue)) > 1e-6) {
                printf("Skipping N = %d for radix %d (not power of radix)\n", fftLength, fftRadix);
                continue;
            }
            fftStage = (int)(round(logValue));
            fftType = DIT;

            complexFloat *input = malloc(sizeof(complexFloat) * fftLength);
            complexFloat *output = malloc(sizeof(complexFloat) * fftLength);

            for (int j = 0; j < fftLength; ++j) {
                input[j].re = cosf(2 * M_PI * 4 * j / fftLength);
                input[j].im = 0;
            }

            clock_t start = clock();
            fft(input, output, false);
            clock_t end = clock();

            double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
            printf("N = %4d | Time: %.6f s\n", fftLength, elapsed);

            free(input);
            free(output);
        }
    }
    return 0;
}
