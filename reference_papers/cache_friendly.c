// Corrected Iterative Radix-4 FFT with float-only representation, RMS comparison to Radix-2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float *cos_table = NULL;
float *sin_table = NULL;

void precompute_twiddles(int N) {
    cos_table = malloc((N / 2) * sizeof(float));
    sin_table = malloc((N / 2) * sizeof(float));
    for (int i = 0; i < N / 2; ++i) {
        float angle = -2.0f * M_PI * i / N;
        cos_table[i] = cosf(angle);
        sin_table[i] = sinf(angle);
    }
}

void bit_reverse_radix4(float *re, float *im, int N) {
    int bits = log2(N) / 2;
    for (int i = 0; i < N; ++i) {
        int j = 0, x = i;
        for (int b = 0; b < bits; ++b) {
            j = (j << 2) | (x & 3);
            x >>= 2;
        }
        if (j > i) {
            float temp_re = re[i], temp_im = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = temp_re; im[j] = temp_im;
        }
    }
}

void radix4_fft_iterative(float *re, float *im, int N) {
    bit_reverse_radix4(re, im, N);
    for (int len = 4; len <= N; len *= 4) {
        int step = N / len;
        for (int i = 0; i < N; i += len) {
            for (int j = 0; j < len / 4; ++j) {
                int idx1 = j * step;
                int idx2 = 2 * idx1;
                int idx3 = 3 * idx1;

                float w1_re = cos_table[idx1], w1_im = sin_table[idx1];
                float w2_re = cos_table[idx2], w2_im = sin_table[idx2];
                float w3_re = cos_table[idx3], w3_im = sin_table[idx3];

                int a = i + j;
                int b = a + len / 4;
                int c = b + len / 4;
                int d = c + len / 4;

                float ar = re[a], ai = im[a];
                float br = re[b], bi = im[b];
                float cr = re[c], ci = im[c];
                float dr = re[d], di = im[d];

                float btr = br * w1_re - bi * w1_im;
                float bti = br * w1_im + bi * w1_re;
                float ctr = cr * w2_re - ci * w2_im;
                float cti = cr * w2_im + ci * w2_re;
                float dtr = dr * w3_re - di * w3_im;
                float dti = dr * w3_im + di * w3_re;

                float t0r = ar + ctr;
                float t0i = ai + cti;
                float t1r = ar - ctr;
                float t1i = ai - cti;
                float t2r = btr + dtr;
                float t2i = bti + dti;
                float t3r = btr - dtr;
                float t3i = bti - dti;

                re[a] = t0r + t2r; im[a] = t0i + t2i;
                re[b] = t1r + t3i; im[b] = t1i - t3r;
                re[c] = t0r - t2r; im[c] = t0i - t2i;
                re[d] = t1r - t3i; im[d] = t1i + t3r;
            }
        }
    }
}

void bit_reverse(float *re, float *im, int N) {
    int j = 0;
    for (int i = 0; i < N; ++i) {
        if (i < j) {
            float temp_re = re[i], temp_im = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = temp_re; im[j] = temp_im;
        }
        int m = N >> 1;
        while (j >= m && m > 0) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

void radix2_fft_float(float *re, float *im, int N) {
    bit_reverse(re, im, N);
    for (int s = 1; s <= log2(N); ++s) {
        int m = 1 << s;
        int step = N / m;
        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < m / 2; ++j) {
                int idx = j * step;
                float w_re = cos_table[idx];
                float w_im = sin_table[idx];

                int t = k + j;
                int u = k + j + m / 2;

                float tre = w_re * re[u] - w_im * im[u];
                float tim = w_re * im[u] + w_im * re[u];

                float ure = re[t];
                float uim = im[t];

                re[t] = ure + tre;
                im[t] = uim + tim;
                re[u] = ure - tre;
                im[u] = uim - tim;
            }
        }
    }
}

void print_fft_float(float *re, float *im, int N, const char *label) {
    printf("%s:\n", label);
    for (int i = 0; i < N; ++i)
        printf("X[%d] = %.5f + %.5fi\n", i, re[i], im[i]);
}

double compute_rms_error(float *re1, float *im1, float *re2, float *im2, int N) {
    double err = 0.0;
    for (int i = 0; i < N; ++i) {
        double dr = re1[i] - re2[i];
        double di = im1[i] - im2[i];
        err += dr * dr + di * di;
    }
    return sqrt(err / N);
}

int main() {
    int N = 64*64;
    float *re = malloc(N * sizeof(float));
    float *im = calloc(N, sizeof(float));

    for (int i = 0; i < N; ++i)
        re[i] = cosf(2 * M_PI * 4 * i / N);

    float *re_r2 = malloc(N * sizeof(float));
    float *im_r2 = calloc(N, sizeof(float));
    float *re_r4 = malloc(N * sizeof(float));
    float *im_r4 = calloc(N, sizeof(float));

    for (int i = 0; i < N; ++i) {
        re_r2[i] = re[i];
        re_r4[i] = re[i];
    }

    precompute_twiddles(N);

    clock_t t1 = clock();
    radix2_fft_float(re_r2, im_r2, N);
    clock_t t2 = clock();

    clock_t t3 = clock();
    radix4_fft_iterative(re_r4, im_r4, N);
    clock_t t4 = clock();

    //print_fft_float(re_r2, im_r2, N, "Radix-2 FFT");
    //print_fft_float(re_r4, im_r4, N, "Radix-4 FFT");

    printf("Execution Time (Radix-2 FFT): %.6f s\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
    printf("Execution Time (Radix-4 FFT): %.6f s\n", (double)(t4 - t3) / CLOCKS_PER_SEC);

    double rms = compute_rms_error(re_r2, im_r2, re_r4, im_r4, N);
    printf("RMS Error (Radix-2 vs Radix-4): %.10e\n", rms);

    free(re); free(im);
    free(re_r2); free(im_r2);
    free(re_r4); free(im_r4);
    free(cos_table); free(sin_table);
    return 0;
}