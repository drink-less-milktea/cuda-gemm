#pragma once
#include "../common/utils.cuh"

void naiveSgemmLauncher(float *a, float *b, float *c, int M, int N, int K);

__global__ void naiveSgemm(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int N, int K);
