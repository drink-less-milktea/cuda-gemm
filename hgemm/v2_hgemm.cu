#include "v2_hgemm.cuh"

using namespace nvcuda;

void v2HgemmLauncher(half *a, half *b, half *c, int M, int N, int K) {
    constexpr int BM = 128, BN = 256;
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    v2Hgemm<<<grid, block>>>(a, b, c, M, N, K);
}

__global__ void v2Hgemm(half *__restrict__ a, half *__restrict__ b,
                        half *__restrict__ c, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32, APAD = 8, BPAD = 8;
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x, wid = tid >> 5;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
        frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
        frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(frag_c[i][j], 0.0);

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    // Pre-compute shared memory byte offsets for cp.async
    int s_a_base = __cvta_generic_to_shared(s_a[0]);
    int s_b_base = __cvta_generic_to_shared(s_b[0]);
    int ld_a = (BK + APAD) * sizeof(half);
    int ld_b = (BN + BPAD) * sizeof(half);
    int s_a_addr0 = s_a_base + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int s_a_addr1 = s_a_addr0 + ld_a;
    int s_b_addr0 = s_b_base + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int s_b_addr1 = s_b_addr0 + ld_b;
    int s_b_addr2 = s_b_addr0 + 2 * ld_b;
    int s_b_addr3 = s_b_addr0 + 3 * ld_b;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; ++bk) {
        // cp.async.ca: asynchronous copy global→shared, cache-all hint
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"(s_a_addr0), "l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"(s_a_addr1), "l"(&a[load_a_gmem_addr + K]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"(s_b_addr0), "l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"(s_b_addr1), "l"(&b[load_b_gmem_addr + N]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"(s_b_addr2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"(s_b_addr3), "l"(&b[load_b_gmem_addr + 3 * N]));

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        // WMMA loads + MMA (same pattern as v1)
        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        __syncthreads();
    }

    int store_m = by * BM + comp_c_frag_m * 64;
    int store_n = bx * BN + comp_c_frag_n * 64;
    int store_addr = OFFSET(store_m, store_n, N);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            wmma::store_matrix_sync(&c[store_addr + i * 16 * N + j * 16],
                                    frag_c[i][j], N, wmma::mem_row_major);
}
