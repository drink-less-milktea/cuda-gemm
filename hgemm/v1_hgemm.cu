#include "v1_hgemm.cuh"

using namespace nvcuda;

void v1HgemmLauncher(half *a, half *b, half *c, int M, int N, int K) {
    constexpr int BM = 128, BN = 256;
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    v1Hgemm<<<grid, block>>>(a, b, c, M, N, K);
}

__global__ void v1Hgemm(half *__restrict__ a, half *__restrict__ b,
                        half *__restrict__ c, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32;
    constexpr int APAD = 8, BPAD = 8;  // padding to avoid bank conflicts

    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x, wid = tid >> 5;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    // WMMA fragments: 2 k-stages × 4 tiles, 4×4 accumulator grid
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
        frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
        frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(frag_c[i][j], 0.0);

    // Thread-to-element mapping for global→shared copies
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // Warp assignment: 2 warps cover M axis, 4 cover N axis
    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; ++bk) {
        // Vectorized loads: 2 rows of A (float4 = 8 halfs), 4 rows of B
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) =
            FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) =
            FLOAT4(a[load_a_gmem_addr + K]);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) =
            FLOAT4(b[load_b_gmem_addr]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) =
            FLOAT4(b[load_b_gmem_addr + N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) =
            FLOAT4(b[load_b_gmem_addr + N * 2]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) =
            FLOAT4(b[load_b_gmem_addr + N * 3]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
        __syncthreads();

        // Load A fragments (2 k-stages, 4 m-tiles each)
        wmma::load_matrix_sync(frag_a[0][0],
                               &s_a[comp_c_frag_m * 64][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1],
                               &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2],
                               &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3],
                               &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0],
                               &s_a[comp_c_frag_m * 64][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1],
                               &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2],
                               &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3],
                               &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        // Load B fragments (2 k-stages, 4 n-tiles each)
        wmma::load_matrix_sync(frag_b[0][0],
                               &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1],
                               &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2],
                               &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3],
                               &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0],
                               &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1],
                               &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2],
                               &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3],
                               &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        // 2 k-stages of MMA accumulate into 4×4 result fragments
#pragma unroll
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        __syncthreads();
    }

    // Store 4×4 tile (64×64 elements) back to global memory
    int store_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_addr = OFFSET(store_gmem_m, store_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            wmma::store_matrix_sync(
                &c[store_addr + i * 16 * N + j * 16], frag_c[i][j], N,
                wmma::mem_row_major);
}
