#include <stdio.h>
#include <immintrin.h>

int main() {
#if defined(__AVX512F__)
    printf("AVX-512 is supported (compiled with AVX512F)\n");
#elif defined(__AVX2__)
    printf("AVX2 is supported (compiled with AVX2)\n");
#elif defined(__AVX__)
    printf("AVX is supported (compiled with AVX)\n");
#else
    printf("No AVX support detected at compile time.\n");
#endif

    return 0;
}