// This code performs a 2D FFT on an input matrix based on the provided filename.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#define PI 3.14159265358979323846

// Function to perform FFT on a 1D array (https://www.youtube.com/watch?app=desktop&v=h7apO7q16V0&t=484s)
void fft(double complex *data, int n) {
    if (n <= 1) return;

    int half = n / 2;
    double complex *even = malloc(half * sizeof(double complex));
    double complex *odd  = malloc(half * sizeof(double complex));
    if (!even || !odd) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Divide
    for (int i = 0; i < half; i++) {
        even[i] = data[i * 2];
        odd[i]  = data[i * 2 + 1];
    }

    // Conquer
    fft(even, half);
    fft(odd, half);

    // Combine
    for (int k = 0; k < half; k++) {
        double complex omega = cexp(-2.0 * PI * I * k / n);
        data[k]         = even[k] + omega * odd[k];
        data[k + half]  = even[k] - omega * odd[k];
    }

    free(even);
    free(odd);
}

// Function to read the matrix input data
// The file is expected to have an input like:
// 1 2
// 3 4
int readMatrix(const char *filename, double complex **data, int *height, int *width) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return 0;
    }

    char line[1024];
    int h = 0, w = 0;
    char *token;

    while (fgets(line, sizeof(line), file) != NULL) {
        int allSpace = 1;
        for (int i = 0; line[i]; i++) {
            if (line[i] != ' ' && line[i] != '\t' && line[i] != '\n') {
                allSpace = 0;
                break;
            }
        }
        if (allSpace) continue;
        int count = 0;
        char lineCopy[1024];
        strcpy(lineCopy, line);
        token = strtok(lineCopy, " \t\n");
        while (token) {
            count++;
            token = strtok(NULL, " \t\n");
        }
        if (count == 0) continue;
        if (h == 0) {
            w = count;
        } else {
            if (count != w) {
                fprintf(stderr, "Error: Inconsistent number of columns in matrix rows.\n");
                fclose(file);
                return 0;
            }
        }
        h++;
    }

    if (h == 0 || w == 0) {
        fprintf(stderr, "Error: No data found in file.\n");
        fclose(file);
        return 0;
    }

    *height = h;
    *width = w;
    *data = malloc(h * w * sizeof(double complex));
    if (!*data) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(file);
        return 0;
    }

    rewind(file);
    int idx = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        int allSpace = 1;
        for (int i = 0; line[i]; i++) {
            if (line[i] != ' ' && line[i] != '\t' && line[i] != '\n') {
                allSpace = 0;
                break;
            }
        }
        if (allSpace) continue;
        token = strtok(line, " \t\n");
        while (token) {
            double val = atof(token);
            (*data)[idx++] = val + 0.0 * I; // use real numbers with an imaginary part of zero.
            token = strtok(NULL, " \t\n");
        }
    }

    fclose(file);
    return 1;
}

// Function to perform 2D FFT (https://namelessalgorithm.com/fft2d/)
int FFT2D(double complex *data, int height, int width) {
    // Perform FFT on each row
    double complex *row = malloc(width * sizeof(double complex));
    if (!row) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            row[j] = data[i * width + j];
        }
        fft(row, width);
        for (int j = 0; j < width; j++) {
            data[i * width + j] = row[j];
        }
    }
    free(row);

    // Perform FFT on each column
    double complex *col = malloc(height * sizeof(double complex));
    if (!col) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            col[i] = data[i * width + j];
        }
        fft(col, height);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = col[i];
        }
    }
    free(col);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file_with_matrix>\n", argv[0]);
        return 1;
    }

    // Read the matrix input data
    double complex *data;
    int height, width;
    if (!readMatrix(argv[1], &data, &height, &width)) {
        return 1;
    }

    if (((height * width) & ((height * width) - 1)) != 0) {
        fprintf(stderr, "Warning: Total number of data points is not a power of 2.\n");
    }

    // Perform the 2D FFT
    FFT2D(data, height, width);

    // Output the results
    for (int i = 0; i < height * width; i++) {
        printf("%lf %lf\n", creal(data[i]), cimag(data[i]));
    }

    free(data);
    return 0;
}