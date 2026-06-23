#include "v3_sgemm.cuh"

void v3SgemmLauncher(float *a, float *b, float *c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    v3Sgemm<<<grid, block>>>(a, b, c, M, N, K);
}

__global__ void v3Sgemm(float *__restrict__ a, float *__restrict__ b,
                        float *__restrict__ c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    // Double-buffering: two copies of shared memory tiles
    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4], r_load_b[4];
    float r_comp_a[TM], r_comp_b[TN];
    float r_c[TM][TN] = {{0.0f}};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // Pre-load first tile (buffer 0)
    {
        int ka = load_a_smem_k, kb = load_b_smem_k;
        FLOAT4(r_load_a[0]) = FLOAT4(a[OFFSET(load_a_gmem_m, ka, K)]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[OFFSET(kb, load_b_gmem_n, N)]);
        s_a[0][ka + 0][load_a_smem_m] = r_load_a[0];
        s_a[0][ka + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][ka + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][ka + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][kb][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();
    }

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1;       // buffer to compute from
        int smem_sel_next = bk & 1;         // buffer to load into

        // Async load next tile
        int ka = bk * BK + load_a_smem_k;
        int kb = bk * BK + load_b_smem_k;
        FLOAT4(r_load_a[0]) = FLOAT4(a[OFFSET(load_a_gmem_m, ka, K)]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[OFFSET(kb, load_b_gmem_n, N)]);

        // Compute using current buffer while next buffer is being loaded
#pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            FLOAT4(r_comp_a[0]) =
                FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) =
                FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) =
                FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) =
                FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
#pragma unroll
            for (int tm = 0; tm < TM; ++tm)
#pragma unroll
                for (int tn = 0; tn < TN; ++tn)
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }

        // Write loaded data into next buffer
        s_a[smem_sel_next][ka - bk * BK + 0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][ka - bk * BK + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][ka - bk * BK + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][ka - bk * BK + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][kb - bk * BK][load_b_smem_n]) =
            FLOAT4(r_load_b[0]);
        __syncthreads();
    }

    // Process last tile (buffer 1)
#pragma unroll
    for (int tk = 0; tk < BK; ++tk) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);
#pragma unroll
        for (int tm = 0; tm < TM; ++tm)
#pragma unroll
            for (int tn = 0; tn < TN; ++tn)
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
    }

    // Write-back
#pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int gm0 = by * BM + ty * TM / 2 + i;
        int gm1 = gm0 + BM / 2;
        int gn = bx * BN + tx * TN / 2;
        FLOAT4(c[OFFSET(gm0, gn, N)]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[OFFSET(gm0, gn + BN / 2, N)]) = FLOAT4(r_c[i][4]);
        FLOAT4(c[OFFSET(gm1, gn, N)]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[OFFSET(gm1, gn + BN / 2, N)]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
