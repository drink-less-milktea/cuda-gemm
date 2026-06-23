#include "common/benchmark.cuh"
#include "sgemm/naive_sgemm.cuh"
#include "sgemm/v1_sgemm.cuh"
#include "sgemm/v2_sgemm.cuh"
#include "sgemm/v3_sgemm.cuh"
#include "hgemm/v1_hgemm.cuh"
#include "hgemm/v2_hgemm.cuh"
#include "hgemm/v3_hgemm.cuh"
#include "hgemm/v4_hgemm.cuh"

#include <cstring>
#include <cstdio>

// Profile a single kernel: print GFLOPS (warmup excluded from nsys capture)
template <typename T>
void profileOne(int M, int N, int K,
                typename LauncherTraits<T>::Launcher launcher,
                const char *label) {
    // Allocate once
    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(T);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(T);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(T);

    T *h_a = static_cast<T *>(std::malloc(a_bytes));
    T *h_b = static_cast<T *>(std::malloc(b_bytes));
    initMatrix(h_a, M, K, 1.0f);
    initMatrix(h_b, K, N, 2.0f);

    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);
    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

    for (int w = 0; w < 5; ++w) {
        cudaMemset(d_c, 0, c_bytes);
        launcher(d_a, d_b, d_c, M, N, K);
        cudaDeviceSynchronize();
    }

    // Mark nsys range start

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_c, 0, c_bytes);
    cudaEventRecord(start);
    launcher(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = calculateGFLOPS(M, N, K, ms / 1000.0);

    // Mark nsys range end

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    std::free(h_a); std::free(h_b);

    std::printf("%-20s  %.3f ms  %.2f GFLOPS\n", label, ms, gflops);
}

int main(int argc, char **argv) {
    int M = 4096;
    if (argc > 1) M = std::atoi(argv[1]);
    int N = M, K = M;

    std::printf("Profiling M=N=K=%d\n\n", M);
    std::printf("%-20s  %-10s  %s\n", "Kernel", "Time(ms)", "GFLOPS");
    std::printf("----------------------------------------------\n");

    // SGEMM
    profileOne<float>(M, N, K, naiveSgemmLauncher, "naive_sgemm");
    profileOne<float>(M, N, K, v1SgemmLauncher, "v1_sgemm");
    profileOne<float>(M, N, K, v2SgemmLauncher, "v2_sgemm");
    profileOne<float>(M, N, K, v3SgemmLauncher, "v3_sgemm");

    // HGEMM
    profileOne<half>(M, N, K, v1HgemmLauncher, "v1_hgemm");
    profileOne<half>(M, N, K, v2HgemmLauncher, "v2_hgemm");
    profileOne<half>(M, N, K, v3HgemmLauncher, "v3_hgemm");
    profileOne<half>(M, N, K, v4HgemmLauncher, "v4_hgemm");

    return 0;
}
