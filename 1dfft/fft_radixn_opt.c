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
//static inline digit_reverse_radix4(double complex *data, int n);
void fft_radix4(double complex *data, int n);
double compute_rms_error(double complex *a, double complex *b, int n);
void compute_twiddle_factors(double complex *W1, double complex *W2, double complex *W3, int N);
void fft_radix4_optimized(double complex *x, int N);
int next_power_of_4(int N);


static inline void digit_reverse_radix4(double complex *x, int n) {
    int log4n = log2(n) / 2;
    for (int i = 0; i < n; ++i) {
        int rev = 0, x_ = i;
        for (int j = 0; j < log4n; ++j) {
            rev = (rev << 2) | (x_ & 3);
            x_ >>= 2;
        }
        if (i < rev) {
            double complex tmp = x[i];
            x[i] = x[rev];
            x[rev] = tmp;
        }
    }
}

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



int main() {
    double freq = 4.0;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        double value = cos(2 * PI * freq * i / INPUT_SIZE);
        x_cos[i] = value + 0.0 * I;
        x_2[i] = x_cos[i];
        x_4[i] = x_cos[i];
    }
    float final_answer; 
    final_answer = wakeup_delay();

    double time_stamp[2]={0.0,0.0};
    struct timespec time_start, time_stop;
    clock_gettime(CLOCK_REALTIME, &time_start);
    fft_radix2(x_2, INPUT_SIZE);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[0] =  interval(time_start, time_stop);
    clock_gettime(CLOCK_REALTIME, &time_start);
    fft_radix4_optimized(x_4, INPUT_SIZE);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[1] =  interval(time_start, time_stop);

   /* printf("\nFFT Radix-2 Output:\n");
    for (int i = 0; i < INPUT_SIZE; ++i)
        printf("X[%d] = %.5f + %.5fi\n", i, creal(x_2[i]), cimag(x_2[i]));

    printf("\nFFT Radix-4 Output:\n");
    for (int i = 0; i < INPUT_SIZE; ++i)
        printf("X[%d] = %.5f + %.5fi\n", i, creal(x_4[i]), cimag(x_4[i]));*/

    double rms = compute_rms_error(x_2, x_4, INPUT_SIZE);
    printf("\nRMS Error between Radix-2 and Radix-4: %.10e\n", rms);
    printf("Time taken for Radix-2 FFT: %.10f seconds\n", time_stamp[0]);
    printf("Time taken for Radix-4 FFT: %.10f seconds\n", time_stamp[1]);

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


void compute_twiddle_factors(double complex *W1, double complex *W2, double complex *W3, int N) {
    for (int k = 0; k < N / 4; ++k) {
        double angle = -2.0 * PI * k / N;
        W1[k] = cexp(I * angle);
        W2[k] = cexp(I * 2.0 * angle);
        W3[k] = cexp(I * 3.0 * angle);
    }
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



/*
void fft_radix4_optimized(double complex *x, int n) {
    digit_reverse_radix4(x, n);

    int stages = log2(n) / 2;
    for (int s = 0; s < stages; ++s) {
        int m = 1 << (2 * (s + 1)); // m = 4^(s+1)
        int m4 = m >> 2;

        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m4; ++j) {
                double angle = -2.0 * PI * j / m;
                double complex W1 = cexp(I * angle);
                double complex W2 = cexp(I * 2.0 * angle);
                double complex W3 = cexp(I * 3.0 * angle);

                double complex a = x[k + j + 0 * m4];
                double complex b = x[k + j + 1 * m4] * W1;
                double complex c = x[k + j + 2 * m4] * W2;
                double complex d = x[k + j + 3 * m4] * W3;

                double complex A = a + c;
                double complex B = a - c;
                double complex C = b + d;
                double complex D = I * (b - d);

                x[k + j + 0 * m4] = A + C;
                x[k + j + 1 * m4] = B - D;
                x[k + j + 2 * m4] = A - C;
                x[k + j + 3 * m4] = B + D;
            }
        }
    }
}*/
void fft_radix4_optimized(double complex *x, int n) {
    digit_reverse_radix4(x, n);

    int stages = log2(n) / 2;

    for (int s = 0; s < stages; ++s) {
        int m = 1 << (2 * (s + 1)); // m = 4^(s+1)
        int m4 = m >> 2;

        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m4; ++j) {
                // Compute twiddles as real sin/cos
                double angle = -2.0 * PI * j / m;
                double cos1 = cos(angle), sin1 = sin(angle);
                double cos2 = cos(2 * angle), sin2 = sin(2 * angle);
                double cos3 = cos(3 * angle), sin3 = sin(3 * angle);

                // Load inputs as real/imag
                double ar = creal(x[k + j + 0 * m4]), ai = cimag(x[k + j + 0 * m4]);
                double br = creal(x[k + j + 1 * m4]), bi = cimag(x[k + j + 1 * m4]);
                double cr = creal(x[k + j + 2 * m4]), ci = cimag(x[k + j + 2 * m4]);
                double dr = creal(x[k + j + 3 * m4]), di = cimag(x[k + j + 3 * m4]);

                // Twiddle multiplications (complex mult using real math)
                double br_tw = br * cos1 - bi * sin1;
                double bi_tw = br * sin1 + bi * cos1;
                double cr_tw = cr * cos2 - ci * sin2;
                double ci_tw = cr * sin2 + ci * cos2;
                double dr_tw = dr * cos3 - di * sin3;
                double di_tw = dr * sin3 + di * cos3;

                // Butterfly computations
                double Ar = ar + cr_tw;
                double Ai = ai + ci_tw;
                double Br = ar - cr_tw;
                double Bi = ai - ci_tw;
                double Cr = br_tw + dr_tw;
                double Ci = bi_tw + di_tw;
                double Dr = bi_tw - di_tw; // imag(b - d)
                double Di = dr_tw - br_tw; // real(d - b)

                // Combine
                x[k + j + 0 * m4] = (Ar + Cr) + I * (Ai + Ci);
                x[k + j + 1 * m4] = (Br - Dr) + I * (Bi + Di);
                x[k + j + 2 * m4] = (Ar - Cr) + I * (Ai - Ci);
                x[k + j + 3 * m4] = (Br + Dr) + I * (Bi - Di);
            }
        }
    }
}