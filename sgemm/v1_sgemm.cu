#include "v1_sgemm.cuh"

void v1SgemmLauncher(float *a, float *b, float *c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    v1Sgemm<<<grid, block>>>(a, b, c, M, N, K);
}

__global__ void v1Sgemm(float *__restrict__ a, float *__restrict__ b,
                        float *__restrict__ c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {{0.0f}};

    // Global→shared tiling: each thread loads 4 elements (float4)
    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

#pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        // Load A tile (128×8) and B tile (8×128) into shared memory
        int ka = bk * BK + load_a_smem_k;
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) =
            FLOAT4(a[OFFSET(load_a_gmem_m, ka, K)]);
        int kb = bk * BK + load_b_smem_k;
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) =
            FLOAT4(b[OFFSET(kb, load_b_gmem_n, N)]);
        __syncthreads();

        // Each thread computes a TM×TN sub-tile from shared memory
#pragma unroll
        for (int k = 0; k < BK; ++k)
#pragma unroll
            for (int m = 0; m < TM; ++m)
#pragma unroll
                for (int n = 0; n < TN; ++n)
                    r_c[m][n] +=
                        s_a[ty * TM + m][k] * s_b[k][tx * TN + n];
        __syncthreads();
    }

    // Write TM×TN results back to global memory (vectorised float4 stores)
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = by * BM + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int gn = bx * BN + tx * TN + j;
            FLOAT4(c[OFFSET(gm, gn, N)]) = FLOAT4(r_c[i][j]);
        }
    }
}
