#pragma once
#include "../common/utils.cuh"
void v2SgemmLauncher(float *a, float *b, float *c, int M, int N, int K);
__global__ void v2Sgemm(float *__restrict__ a, float *__restrict__ b,
                        float *__restrict__ c, int M, int N, int K);
