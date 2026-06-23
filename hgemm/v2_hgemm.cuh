#pragma once
#include "../common/utils.cuh"
void v2HgemmLauncher(half *a, half *b, half *c, int M, int N, int K);
__global__ void v2Hgemm(half *__restrict__ a, half *__restrict__ b,
                        half *__restrict__ c, int M, int N, int K);
