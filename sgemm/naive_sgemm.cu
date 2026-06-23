#include "naive_sgemm.cuh"

void naiveSgemmLauncher(float *a, float *b, float *c, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naiveSgemm<<<grid, block>>>(a, b, c, M, N, K);
}

__global__ void naiveSgemm(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k)
        sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    c[OFFSET(m, n, N)] = sum;
}
