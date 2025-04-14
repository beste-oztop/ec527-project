#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define FFT_SIZE 64  // Must be a power of 4

typedef struct {
    float real;
    float imag;
} Complex;

void print_fft(Complex* X, int N) {
    for (int k = 0; k < N; ++k) {
        printf("X[%d] = %.5f + %.5fi\n", k, X[k].real, X[k].imag);
    }
}
unsigned int reverse_bits(unsigned int x, int bits) {
    unsigned int y = 0;
    for (int i = 0; i < bits; ++i) {
        y <<= 1;
        y |= (x & 1);
        x >>= 1;
    }
    return y;
}

void bit_reversal_reorder(Complex* X, int N) {
    int bits = 0;
    while ((1 << bits) < N) ++bits;

    for (int i = 0; i < N; ++i) {
        int j = reverse_bits(i, bits);
        if (i < j) {
            Complex tmp = X[i];
            X[i] = X[j];
            X[j] = tmp;
        }
    }
}
void zero_pad(float* x, int len, float* padded, int N) {
    for (int i = 0; i < N; ++i) {
        padded[i] = (i < len) ? x[i] : 0.0f;
    }
}

void compute_twiddle(int M, int scale, Complex* W) {
    for (int k = 0; k < M / 4; ++k) {
        float angle = -2.0f * PI * k * scale / M;
        W[k].real = cosf(angle);
        W[k].imag = sinf(angle);
    }
}

void compute_dft(float* x, Complex* X, int size) {
    for (int k = 0; k < size; ++k) {
        X[k].real = 0;
        X[k].imag = 0;
        for (int n = 0; n < size; ++n) {
            float angle = -2 * PI * k * n / size;
            X[k].real += x[n] * cosf(angle);
            X[k].imag += x[n] * sinf(angle);
        }
    }
}

void radix4_fft(float* x, Complex* X, int N) {
    float xin[N];
    zero_pad(x, N, xin, N);

    float xo[N/4], x1[N/4], x2[N/4], x3[N/4];
    for (int i = 0; i < N/4; ++i) {
        xo[i] = xin[i*4];
        x1[i] = xin[i*4 + 1];
        x2[i] = xin[i*4 + 2];
        x3[i] = xin[i*4 + 3];
    }

    Complex Y[N/4], Z[N/4], G[N/4], H[N/4];
    Complex W1[N/4], W2[N/4], W3[N/4];

    compute_twiddle(N, 1, W1);
    compute_twiddle(N, 2, W2);
    compute_twiddle(N, 3, W3);

    compute_dft(xo, Y, N / 4);
    compute_dft(x1, Z, N / 4);
    compute_dft(x2, G, N / 4);
    compute_dft(x3, H, N / 4);

    for (int k = 0; k < N/4; ++k) {
        float zr = Z[k].real * W1[k].real - Z[k].imag * W1[k].imag;
        float zi = Z[k].real * W1[k].imag + Z[k].imag * W1[k].real;
        Z[k].real = zr;
        Z[k].imag = zi;

        float gr = G[k].real * W2[k].real - G[k].imag * W2[k].imag;
        float gi = G[k].real * W2[k].imag + G[k].imag * W2[k].real;
        G[k].real = gr;
        G[k].imag = gi;

        float hr = H[k].real * W3[k].real - H[k].imag * W3[k].imag;
        float hi = H[k].real * W3[k].imag + H[k].imag * W3[k].real;
        H[k].real = hr;
        H[k].imag = hi;

        Complex x_1, x_2, x_3, x_4;

        x_1.real = Y[k].real + Z[k].real + G[k].real + H[k].real;
        x_1.imag = Y[k].imag + Z[k].imag + G[k].imag + H[k].imag;

        x_2.real = Y[k].real - G[k].real - (Z[k].imag - H[k].imag);
        x_2.imag = Y[k].imag - G[k].imag + (Z[k].real - H[k].real);

        x_3.real = Y[k].real - Z[k].real + G[k].real - H[k].real;
        x_3.imag = Y[k].imag - Z[k].imag + G[k].imag - H[k].imag;

        x_4.real = Y[k].real - G[k].real + (Z[k].imag - H[k].imag);
        x_4.imag = Y[k].imag - G[k].imag - (Z[k].real - H[k].real);

        X[k] = x_1;
        X[k + N/4] = x_2;
        X[k + 2*N/4] = x_3;
        X[k + 3*N/4] = x_4;
    }
}

void radix4_digit_reverse(Complex* data, int N) {
    int log4_N = 0;
    int temp_N = N;
    while (temp_N > 1) {
        temp_N /= 4;
        log4_N++;
    }

    for (int i = 0; i < N; ++i) {
        int reversed = 0;
        int temp = i;

        for (int j = 0; j < log4_N; ++j) {
            reversed = reversed * 4 + (temp % 4); // base-4 digit reversal
            temp /= 4;
        }

        if (i < reversed) {
            Complex tmp = data[i];
            data[i] = data[reversed];
            data[reversed] = tmp;
        }
    }
}

float rms_error(Complex* a, Complex* b, int N) {
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        float dr = a[i].real - b[i].real;
        float di = a[i].imag - b[i].imag;
        sum += dr * dr + di * di;
    }
    return sqrtf(sum / N);
}

int main() {
    int N = FFT_SIZE;
    float signal[N];
    Complex X_dft[N], X_fft[N];

    for (int t = 0; t < N; ++t) {
        signal[t] = cosf(2 * PI * 8 * t / N);
    }

    printf("Input Signal:\n");
    for (int t = 0; t < N; ++t) {
        printf("signal[%d] = %.5f\n", t, signal[t]);
    }

    clock_t start_dft = clock();
    compute_dft(signal, X_dft, N);
    clock_t end_dft = clock();
    double time_dft = (double)(end_dft - start_dft) / CLOCKS_PER_SEC;

    clock_t start_fft = clock();
    radix4_fft(signal, X_fft, N);

    printf("FFT Output Before Reordering:\n");
    print_fft(X_fft, N);

    // Apply radix-4 digit reversal to match DFT ordering
    radix4_digit_reverse(X_fft, N);

    clock_t end_fft = clock();
    double time_fft = (double)(end_fft - start_fft) / CLOCKS_PER_SEC;
    float error = rms_error(X_dft, X_fft, N);

    printf("DFT Time: %.6f sec\n", time_dft);
    printf("Radix-4 FFT Time: %.6f sec\n", time_fft);
    printf("Speedup: %.2f%%\n", (1 - time_fft / time_dft) * 100);
    printf("RMS Error: %.10e\n", error);
    printf("DFT Output:\n");
    print_fft(X_dft, N);
    printf("FFT Output After Reordering:\n");
    print_fft(X_fft, N);

    return 0;
}