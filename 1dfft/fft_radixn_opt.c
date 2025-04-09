#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define INPUT_SIZE 16

double complex x_cos[INPUT_SIZE];
double complex x_2[INPUT_SIZE];
double complex x_4[INPUT_SIZE];

void bit_reverse(double complex *data, int n);
void fft_radix2(double complex *data, int n);
void digit_reverse_radix4(double complex *data, int n);
void fft_radix4(double complex *data, int n);
double compute_rms_error(double complex *a, double complex *b, int n);
void compute_twiddle_factors(double complex *W1, double complex *W2, double complex *W3, int N);
void fft_radix4_nonrecursive(double complex *x, int N);
int next_power_of_4(int N);

int main() {
    double freq = 4.0;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        double value = cos(2 * PI * freq * i / INPUT_SIZE);
        x_cos[i] = value + 0.0 * I;
        x_2[i] = x_cos[i];
        x_4[i] = x_cos[i];
    }

    fft_radix2(x_2, INPUT_SIZE);
    fft_radix4_nonrecursive(x_4, INPUT_SIZE);

    printf("\nFFT Radix-2 Output:\n");
    for (int i = 0; i < INPUT_SIZE; ++i)
        printf("X[%d] = %.5f + %.5fi\n", i, creal(x_2[i]), cimag(x_2[i]));

    printf("\nFFT Radix-4 Output:\n");
    for (int i = 0; i < INPUT_SIZE; ++i)
        printf("X[%d] = %.5f + %.5fi\n", i, creal(x_4[i]), cimag(x_4[i]));

    double rms = compute_rms_error(x_2, x_4, INPUT_SIZE);
    printf("\nRMS Error between Radix-2 and Radix-4: %.10e\n", rms);

    return 0;
}

void bit_reverse(double complex *data, int n) {
    int j = 0;
    for (int i = 0; i < n; ++i) {
        if (i < j) {
            double complex temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        int m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

void fft_radix2(double complex *data, int n) {
    bit_reverse(data, n);
    for (int s = 1; s <= (int)log2((double)n); ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        double complex wm = cexp(-2.0 * PI * I / m);
        for (int k = 0; k < n; k += m) {
            double complex w = 1.0;
            for (int j = 0; j < m2; ++j) {
                double complex t = w * data[k + j + m2];
                double complex u = data[k + j];
                data[k + j] = u + t;
                data[k + j + m2] = u - t;
                w *= wm;
            }
        }
    }
}

void digit_reverse_radix4(double complex *data, int n) {
    int log4n = log2(n) / 2;
    for (int i = 0; i < n; ++i) {
        int rev = 0, x = i;
        for (int j = 0; j < log4n; ++j) {
            rev = (rev << 2) | (x & 3);
            x >>= 2;
        }
        if (i < rev) {
            double complex temp = data[i];
            data[i] = data[rev];
            data[rev] = temp;
        }
    }
}

void compute_twiddle_factors(double complex *W1, double complex *W2, double complex *W3, int N) {
    for (int k = 0; k < N / 4; ++k) {
        double angle = -2.0 * PI * k / N;
        W1[k] = cexp(I * angle);
        W2[k] = cexp(I * 2.0 * angle);
        W3[k] = cexp(I * 3.0 * angle);
    }
}

void fft_radix4_nonrecursive(double complex *x, int N) {
    if (N % 4 != 0) {
        fprintf(stderr, "Input size must be a power of 4.\n");
        exit(EXIT_FAILURE);
    }

    int M = N / 4;

    double complex *xo = malloc(M * sizeof(double complex));
    double complex *x1 = malloc(M * sizeof(double complex));
    double complex *x2 = malloc(M * sizeof(double complex));
    double complex *x3 = malloc(M * sizeof(double complex));

    for (int i = 0; i < M; ++i) {
        xo[i] = x[4 * i + 0];
        x1[i] = x[4 * i + 1];
        x2[i] = x[4 * i + 2];
        x3[i] = x[4 * i + 3];
    }

    // DFT of length M (simple DFT with DFT matrix)
    double complex *A = malloc(M * M * sizeof(double complex));
    for (int k = 0; k < M; ++k) {
        for (int n = 0; n < M; ++n) {
            A[k * M + n] = cexp(-2.0 * PI * I * k * n / M);
        }
    }

    double complex *Y = calloc(M, sizeof(double complex));
    double complex *Z = calloc(M, sizeof(double complex));
    double complex *G = calloc(M, sizeof(double complex));
    double complex *H = calloc(M, sizeof(double complex));

    for (int k = 0; k < M; ++k) {
        for (int n = 0; n < M; ++n) {
            Y[k] += xo[n] * A[k * M + n];
            Z[k] += x1[n] * A[k * M + n];
            G[k] += x2[n] * A[k * M + n];
            H[k] += x3[n] * A[k * M + n];
        }
    }

    // Apply twiddle factors
    double complex *W1 = malloc(M * sizeof(double complex));
    double complex *W2 = malloc(M * sizeof(double complex));
    double complex *W3 = malloc(M * sizeof(double complex));
    compute_twiddle_factors(W1, W2, W3, N);

    for (int k = 0; k < M; ++k) {
        Z[k] *= W1[k];
        G[k] *= W2[k];
        H[k] *= W3[k];
    }

    // Combine results using radix-4 butterflies
    for (int k = 0; k < M; ++k) {
        x[k + 0 * M] = Y[k] + Z[k] + G[k] + H[k];
        x[k + 1 * M] = Y[k] - I * Z[k] - G[k] + I * H[k];
        x[k + 2 * M] = Y[k] - Z[k] + G[k] - H[k];
        x[k + 3 * M] = Y[k] + I * Z[k] - G[k] - I * H[k];
    }

    // Free allocated memory
    free(xo); free(x1); free(x2); free(x3);
    free(Y); free(Z); free(G); free(H);
    free(A); free(W1); free(W2); free(W3);
}

int next_power_of_4(int N) {
    int pow = 1;
    while (pow < N) pow *= 4;
    return pow;
}

double compute_rms_error(double complex *a, double complex *b, int n) {
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff_real = creal(a[i]) - creal(b[i]);
        double diff_imag = cimag(a[i]) - cimag(b[i]);
        err += diff_real * diff_real + diff_imag * diff_imag;
    }
    return sqrt(err / n);
}
