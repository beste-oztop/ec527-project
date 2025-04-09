// C version of MATLAB's DIT_FFT_rad4 function with Radix-2 FFT and DFT comparison
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

int next_power_of_4(int n) {
    int N = 1;
    while (N < n) N <<= 2;
    return N;
}

cplx *zero_pad_radix4(cplx *x, int len, int N) {
    cplx *xin = calloc(N, sizeof(cplx));
    for (int i = 0; i < len; i++) xin[i] = x[i];
    return xin;
}

// Naive DFT for ground truth
void naive_dft(cplx *in, cplx *out, int N) {
    for (int k = 0; k < N; ++k) {
        out[k] = 0;
        for (int n = 0; n < N; ++n) {
            double angle = -2 * M_PI * k * n / N;
            out[k] += in[n] * cexp(I * angle);
        }
    }
}
void radix4_fft_recursive(cplx *x, int N) {
    if (N == 1) return;
    int N4 = N / 4;

    cplx *x0 = malloc(N4 * sizeof(cplx));
    cplx *x1 = malloc(N4 * sizeof(cplx));
    cplx *x2 = malloc(N4 * sizeof(cplx));
    cplx *x3 = malloc(N4 * sizeof(cplx));

    for (int i = 0; i < N4; ++i) {
        x0[i] = x[4 * i];
        x1[i] = x[4 * i + 1];
        x2[i] = x[4 * i + 2];
        x3[i] = x[4 * i + 3];
    }

    radix4_fft_recursive(x0, N4);
    radix4_fft_recursive(x1, N4);
    radix4_fft_recursive(x2, N4);
    radix4_fft_recursive(x3, N4);

    for (int k = 0; k < N4; ++k) {
        double angle1 = -2 * M_PI * k / N;
        double angle2 = -2 * M_PI * 2 * k / N;
        double angle3 = -2 * M_PI * 3 * k / N;
        cplx w1 = cexp(I * angle1);
        cplx w2 = cexp(I * angle2);
        cplx w3 = cexp(I * angle3);

        cplx a0 = x0[k];
        cplx a1 = x1[k] * w1;
        cplx a2 = x2[k] * w2;
        cplx a3 = x3[k] * w3;

        cplx t0 = a0 + a2;
        cplx t1 = a0 - a2;
        cplx t2 = a1 + a3;
        cplx t3 = I * (a1 - a3);

        x[k] = t0 + t2;
        x[k + N4] = t1 + t3;
        x[k + 2 * N4] = t0 - t2;
        x[k + 3 * N4] = t1 - t3;
    }

    free(x0); free(x1); free(x2); free(x3);
}
// Iterative in-place Radix-2 Cooley-Tukey FFT
void radix2_fft(cplx *x, int N) {
    int log2_N = log2(N);
    for (int i = 0, j = 0; i < N; ++i) {
        if (i < j) {
            cplx tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
        int m = N >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    for (int s = 1; s <= log2_N; ++s) {
        int m = 1 << s;
        cplx wm = cexp(-2.0 * I * M_PI / m);
        for (int k = 0; k < N; k += m) {
            cplx w = 1;
            for (int j = 0; j < m / 2; ++j) {
                cplx t = w * x[k + j + m / 2];
                cplx u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
}

// DIT FFT Radix-4
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

    naive_dft(xo, Y, L);
    naive_dft(x1, Z, L);
    naive_dft(x2, G, L);
    naive_dft(x3, H, L);

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

double compute_rms_error(cplx *a, cplx *b, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double re_diff = creal(a[i]) - creal(b[i]);
        double im_diff = cimag(a[i]) - cimag(b[i]);
        sum += re_diff * re_diff + im_diff * im_diff;
    }
    return sqrt(sum / N);
}


int main() {
    int N = 64;
    cplx *signal = malloc(N * sizeof(cplx));
    for (int i = 0; i < N; ++i)
        signal[i] = cos(2 * M_PI * 4 * i / N);

    // Radix-4 FFT
    cplx *x_rad4 = malloc(N * sizeof(cplx));
    for (int i = 0; i < N; ++i) x_rad4[i] = signal[i];
    clock_t t1 = clock();
    radix4_fft_recursive(x_rad4, N);
    clock_t t2 = clock();

    // Radix-2 FFT
    cplx *x_rad2 = malloc(N * sizeof(cplx));
    for (int i = 0; i < N; ++i) x_rad2[i] = signal[i];
    clock_t t3 = clock();
    radix2_fft(x_rad2, N);
    clock_t t4 = clock();

    // Naive DFT
    cplx *x_dft = malloc(N * sizeof(cplx));
    clock_t t5 = clock();
    naive_dft(signal, x_dft, N);
    clock_t t6 = clock();

    double time_r4 = (double)(t2 - t1) / CLOCKS_PER_SEC;
    double time_r2 = (double)(t4 - t3) / CLOCKS_PER_SEC;
    double time_dft = (double)(t6 - t5) / CLOCKS_PER_SEC;

    double err_r4 = compute_rms_error(x_rad4, x_dft, N);
    double err_r2 = compute_rms_error(x_rad2, x_dft, N);

    printf("Execution Time (Radix-4 FFT): %.6f s\n", time_r4);
    printf("Execution Time (Radix-2 FFT): %.6f s\n", time_r2);
    printf("Execution Time (Naive DFT):   %.6f s\n", time_dft);
    printf("RMS Error (Radix-4 vs DFT): %.10e\n", err_r4);
    printf("RMS Error (Radix-2 vs DFT): %.10e\n", err_r2);

    free(signal); free(x_rad4); free(x_rad2); free(x_dft);
    return 0;
}