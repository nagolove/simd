#include <stddef.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

float *data_alloc(int num) {
    return aligned_alloc(32, num * sizeof(float));
}

void data_fill(float *data, int num) {
    for (int i = 0; i < num; i++) {
        data[i] = rand() / (double)RAND_MAX;
    }
}

void data_print(float *data, int num) {
    for (int i = 0; i < num; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
}

float data_sum(float *data, int num) {
    float accum = 0.;
    for (int i = 0; i < num; i++) {
        accum += data[i];
    }
    return accum;
}

float data_sum_unroll(float *data, int num) {
    float accum = 0.;
    int num8 = num / 8;
    int i;
    for (i = 0; i < num8; i += 8) {
        accum += data[i + 0];
        accum += data[i + 1];
        accum += data[i + 2];
        accum += data[i + 3];
        accum += data[i + 4];
        accum += data[i + 5];
        accum += data[i + 6];
        accum += data[i + 7];
    }

    for (i = 0; i < num; i += 8) {
        accum += data[i];
    }

    return accum;
}

float data_sum_sse(float *data, int num) {
    __m128 accum = {0};
    int num8 = num - num % 4;
    int i;

    for (i = 0; i < num8; i += 4) {
        __m128 a;
        a[0] = data[i + 0];
        a[1] = data[i + 1];
        a[2] = data[i + 2];
        a[3] = data[i + 3];

        accum = _mm_add_ss(a, accum);
    }

    float accumf = 0.;
    for (; i < num; i++) {
        accumf += data[i];
    }

    accumf += accum[0];
    accumf += accum[1];
    accumf += accum[2];
    accumf += accum[3];
    return accumf;
}

float data_sum_avx(float *data, int num) {
    __m256 accum = {0.};
    int num8 = num - num % 8;
    int i;

    for (i = 0; i < num8; i += 8) {
        //__m256 a = _mm256_loadu_ps(&data[i]);
        __m256 a = _mm256_load_ps(&data[i]);
        accum = _mm256_add_ps(a, accum);
    }

    float accumf = 0.;
    for (; i < num; i++) {
        accumf += data[i];
    }

    for (int j = 0; j < 8; ++j) 
        accumf += accum[j];
    return accumf;
}

/*
float data_sum_avx512(float *data, int num) {
    __m512 accum = {0};
    int num8 = num - num % 16;
    int i;

    for (i = 0; i < num8; i += 16) {
        __m512 a;
        a[0] = data[i + 0];
        a[1] = data[i + 1];
        a[2] = data[i + 2];
        a[3] = data[i + 3];
        a[4] = data[i + 4];
        a[5] = data[i + 5];
        a[6] = data[i + 6];
        a[7] = data[i + 7];
        a[8] = data[i + 8];
        a[9] = data[i + 9];
        a[10] = data[i + 10];
        a[11] = data[i + 11];
        a[12] = data[i + 12];
        a[13] = data[i + 13];
        a[14] = data[i + 14];
        a[15] = data[i + 15];
        //_mm512_add_ps();
        accum = _mm512_add_ps(a, accum);
    }

    float accumf = 0.;
    for (; i < num; i++) {
        accumf += data[i];
    }

    for (int j = 0; j < 16; ++j) 
        accumf += accum[j];
    return accumf;
}
*/

int main() {
    const int num = 500000007;
    float *data = data_alloc(num);
    srand(time(NULL));
    data_fill(data, num);
    /*data_print(data, num);*/
    clock_t t_start, t_stop;
    float sum;

    t_start = clock();
    sum = data_sum(data, num);
    t_stop = clock();
    printf("sum elapsed: %f sec\n", (t_stop - t_start) / (float)CLOCKS_PER_SEC);
    printf("sum: %f\n", sum);

    t_start = clock();
    sum = data_sum_unroll(data, num);
    t_stop = clock();
    printf("sum_unroll elapsed: %f sec\n", (t_stop - t_start) / (float)CLOCKS_PER_SEC);
    printf("sum: %f\n", sum);

    t_start = clock();
    sum = data_sum_sse(data, num);
    t_stop = clock();
    printf("sum_sse elapsed: %f sec\n", (t_stop - t_start) / (float)CLOCKS_PER_SEC);
    printf("sum: %f\n", sum);

    t_start = clock();
    sum = data_sum_avx(data, num);
    t_stop = clock();
    printf("sum_avx elapsed: %f sec\n", (t_stop - t_start) / (float)CLOCKS_PER_SEC);
    printf("sum: %f\n", sum);

    /*
    t_start = clock();
    sum = data_sum_avx512(data, num);
    t_stop = clock();
    printf("sum_avx512 elapsed: %f sec\n", (t_stop - t_start) / (float)CLOCKS_PER_SEC);
    printf("sum: %f\n", sum);
    */

    free(data);
    return 0;
}
