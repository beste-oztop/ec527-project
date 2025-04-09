// C version of MATLAB's DIT_FFT_rad4 function with Radix-2 DFT comparison
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef double complex cplx;

// Compute twiddle factor W_n^k
cplx *compute_twiddle_radix4(int M, int k_mul) {
    int L = M / 4;
    cplx *W = malloc(L * sizeof(cplx));
    for (int k = 0; k < L; ++k) {
        double angle = -2 * M_PI * k_mul * k / M;
        W[k] = cexp(I * angle);
    }
    return W;
}

// Pad input to next power of 4
int next_power_of_4(int n) {
    int N = 1;
    while (N < n) N <<= 2;
    return N;
}

// Zero-pad signal to length N (must be power of 4)
cplx *zero_pad_radix4(cplx *x, int len, int N) {
    cplx *xin = calloc(N, sizeof(cplx));
    for (int i = 0; i < len; i++) xin[i] = x[i];
    return xin;
}

// Naive DFT for radix-2 reference
void radix2_dft(cplx *in, cplx *out, int N) {
    for (int k = 0; k < N; ++k) {
        out[k] = 0;
        for (int n = 0; n < N; ++n) {
            double angle = -2 * M_PI * k * n / N;
            out[k] += in[n] * cexp(I * angle);
        }
    }
}

// Compute 1D DFT matrix-vector multiplication
void dft_naive(cplx *in, cplx *out, int N) {
    for (int k = 0; k < N; ++k) {
        out[k] = 0;
        for (int n = 0; n < N; ++n) {
            double angle = -2 * M_PI * k * n / N;
            out[k] += in[n] * cexp(I * angle);
        }
    }
}

// Main DIT FFT Radix-4
cplx *DIT_FFT_rad4(cplx *x, int len, int N) {
    if (N == 0) N = next_power_of_4(len);
    if ((N & (N - 1)) != 0 || (int)(log(N)/log(4)) * 2 != (int)log2(N)) {
        fprintf(stderr, "N must be a power of 4\n");
        return NULL;
    }
    cplx *xin = (N >= len) ? zero_pad_radix4(x, len, N) : x;
    int M = N;

    int L = M / 4;
    cplx *xo = malloc(L * sizeof(cplx));
    cplx *x1 = malloc(L * sizeof(cplx));
    cplx *x2 = malloc(L * sizeof(cplx));
    cplx *x3 = malloc(L * sizeof(cplx));

    for (int i = 0; i < L; ++i) {
        xo[i] = xin[4 * i];
        x1[i] = xin[4 * i + 1];
        x2[i] = xin[4 * i + 2];
        x3[i] = xin[4 * i + 3];
    }

    cplx *W1 = compute_twiddle_radix4(M, 1);
    cplx *W2 = compute_twiddle_radix4(M, 2);
    cplx *W3 = compute_twiddle_radix4(M, 3);

    cplx *Y = malloc(L * sizeof(cplx));
    cplx *Z = malloc(L * sizeof(cplx));
    cplx *G = malloc(L * sizeof(cplx));
    cplx *H = malloc(L * sizeof(cplx));

    dft_naive(xo, Y, L);
    dft_naive(x1, Z, L);
    dft_naive(x2, G, L);
    dft_naive(x3, H, L);

    for (int i = 0; i < L; ++i) {
        Z[i] *= W1[i];
        G[i] *= W2[i];
        H[i] *= W3[i];
    }

    cplx *X = malloc(M * sizeof(cplx));
    for (int i = 0; i < L; ++i) {
        X[i] = Y[i] + Z[i] + G[i] + H[i];
        X[i + L] = Y[i] - I * Z[i] - G[i] + I * H[i];
        X[i + 2 * L] = Y[i] - Z[i] + G[i] - H[i];
        X[i + 3 * L] = Y[i] + I * Z[i] - G[i] - I * H[i];
    }

    free(xo); free(x1); free(x2); free(x3);
    free(W1); free(W2); free(W3);
    free(Y); free(Z); free(G); free(H);
    if (xin != x) free(xin);
    return X;
}

void print_fft(cplx *X, int N) {
    for (int i = 0; i < N; ++i)
        printf("X[%d] = %.5f + %.5fi\n", i, creal(X[i]), cimag(X[i]));
}

int main() {
    int N = 64*64;
    cplx *signal = malloc(N * sizeof(cplx));
    for (int i = 0; i < N; ++i)
        signal[i] = cos(2 * M_PI * 4 * i / N);

    clock_t t1 = clock();
    cplx *X_rad4 = DIT_FFT_rad4(signal, N, 0);
    clock_t t2 = clock();

    cplx *X_rad2 = malloc(N * sizeof(cplx));
    clock_t t3 = clock();
    radix2_dft(signal, X_rad2, N);
    clock_t t4 = clock();

    double time_rad4 = (double)(t2 - t1) / CLOCKS_PER_SEC;
    double time_rad2 = (double)(t4 - t3) / CLOCKS_PER_SEC;

    printf("\nRadix-4 DIT FFT Result:\n");
    //print_fft(X_rad4, N);
    printf("\nExecution Time (Radix-4 DIT FFT): %.6f seconds\n", time_rad4);

    printf("\nRadix-2 DFT Result:\n");
    //print_fft(X_rad2, N);
    printf("\nExecution Time (Naive DFT): %.6f seconds\n", time_rad2);

    free(signal);
    free(X_rad4);
    free(X_rad2);
    return 0;
}