#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define INPUT_SIZE 64*64

double complex x_cos[INPUT_SIZE];
double complex x_2[INPUT_SIZE];
double complex x_4[INPUT_SIZE];

void bit_reverse(double complex *data, int n);
void fft_radix2(double complex *data, int n);
void digit_reverse_radix4(double complex *data, int n);
void fft_radix4(double complex *data, int n);
double compute_rms_error(double complex *a, double complex *b, int n);

int main() {
    double freq = 64*62.0;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        double value = cos(2 * PI * freq * i / INPUT_SIZE);
        x_cos[i] = value + 0.0 * I;
        x_2[i] = x_cos[i];
        x_4[i] = x_cos[i];
    }

    fft_radix2(x_2, INPUT_SIZE);
    fft_radix4(x_4, INPUT_SIZE);

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

void fft_radix4(double complex *x, int N) {
    if (N == 1) return;
    if (N % 4 != 0) {
        fprintf(stderr, "Input size must be a power of 4.\n");
        exit(EXIT_FAILURE);
    }

    int M = N / 4;
    double complex *x0 = malloc(M * sizeof(double complex));
    double complex *x1 = malloc(M * sizeof(double complex));
    double complex *x2 = malloc(M * sizeof(double complex));
    double complex *x3 = malloc(M * sizeof(double complex));

    for (int i = 0; i < M; i++) {
        x0[i] = x[i * 4 + 0];
        x1[i] = x[i * 4 + 1];
        x2[i] = x[i * 4 + 2];
        x3[i] = x[i * 4 + 3];
    }

    fft_radix4(x0, M);
    fft_radix4(x1, M);
    fft_radix4(x2, M);
    fft_radix4(x3, M);

    for (int k = 0; k < M; k++) {
        double complex W1 = cexp(-2.0 * PI * I * k / N);
        double complex W2 = cexp(-2.0 * PI * I * 2 * k / N);
        double complex W3 = cexp(-2.0 * PI * I * 3 * k / N);

        double complex A = x0[k];
        double complex B = x1[k] * W1;
        double complex C = x2[k] * W2;
        double complex D = x3[k] * W3;

        x[k + 0*M] = A + B + C + D;
        x[k + 1*M] = A - I*B - C + I*D;
        x[k + 2*M] = A - B + C - D;
        x[k + 3*M] = A + I*B - C - I*D;
    }

    free(x0); free(x1); free(x2); free(x3);
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
