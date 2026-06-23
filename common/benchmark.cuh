#pragma once
#include "utils.cuh"

// ===========================================================================
// Template benchmark / correctness framework for SGEMM (float) and HGEMM (half)
// ===========================================================================

template <typename T>
struct LauncherTraits;

// Both SGEMM and HGEMM launchers now use the same signature:
// launcher(T *a, T *b, T *c, int M, int N, int K)
// Each launcher manages gridDim/blockDim internally.
template <>
struct LauncherTraits<float> {
    using Launcher = void (*)(float *, float *, float *, int, int, int);
    static constexpr const char *name = "SGEMM";
};

template <>
struct LauncherTraits<half> {
    using Launcher = void (*)(half *, half *, half *, int, int, int);
    static constexpr const char *name = "HGEMM";
};

// =========================================================================
// cuBLAS reference
// =========================================================================
namespace detail {

inline bool runCublasSgemm(int M, int N, int K, float *d_a, float *d_b,
                           float *d_c, cublasHandle_t handle, cudaEvent_t start,
                           cudaEvent_t stop, double &gflops,
                           float *&host_row_major) {
    host_row_major = nullptr;
    const float alpha = 1.0f, beta = 0.0f;

    for (int w = 0; w < 20; ++w) {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_b, K,
                    d_a, M, &beta, d_c, N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_b, K, d_a,
                M, &beta, d_c, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    gflops = calculateGFLOPS(M, N, K, ms / 1000.0);

    size_t nbytes = static_cast<size_t>(M) * N * sizeof(float);
    float *col = static_cast<float *>(std::malloc(nbytes));
    cudaMemcpy(col, d_c, nbytes, cudaMemcpyDeviceToHost);
    float *row = static_cast<float *>(std::malloc(nbytes));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            row[i * N + j] = col[j + i * N];
    std::free(col);
    host_row_major = row;
    return true;
}

inline bool runCublasHgemm(int M, int N, int K, half *d_a, half *d_b,
                           half *d_c, cublasHandle_t handle, cudaEvent_t start,
                           cudaEvent_t stop, double &gflops,
                           half *&host_row_major) {
    host_row_major = nullptr;
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    for (int w = 0; w < 20; ++w) {
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_b, K,
                    d_a, M, &beta, d_c, N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_b, K, d_a,
                M, &beta, d_c, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    gflops = calculateGFLOPS(M, N, K, ms / 1000.0);

    size_t nbytes = static_cast<size_t>(M) * N * sizeof(half);
    half *col = static_cast<half *>(std::malloc(nbytes));
    cudaMemcpy(col, d_c, nbytes, cudaMemcpyDeviceToHost);
    half *row = static_cast<half *>(std::malloc(nbytes));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            row[i * N + j] = col[j + i * N];
    std::free(col);
    host_row_major = row;
    return true;
}

} // namespace detail

// =========================================================================
// testGEMM: performance measurement only (returns GFLOPS)
// =========================================================================
template <typename T>
double testGEMM(int M, int N, int K,
                typename LauncherTraits<T>::Launcher launcher) {
    if (launcher == nullptr) {
        // Return cuBLAS baseline
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

        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        double gflops = 0;
        T *dummy = nullptr;
        if constexpr (std::is_same_v<T, float>)
            detail::runCublasSgemm(M, N, K, d_a, d_b, d_c, handle, start, stop,
                                   gflops, dummy);
        else
            detail::runCublasHgemm(M, N, K, d_a, d_b, d_c, handle, start, stop,
                                   gflops, dummy);

        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        std::free(h_a);
        std::free(h_b);
        std::free(dummy);
        return gflops;
    }

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int runs = 10;
    double total_gflops = 0;

    for (int r = 0; r < runs; ++r) {
        cudaMemset(d_c, 0, c_bytes);
        cudaEventRecord(start);
        launcher(d_a, d_b, d_c, M, N, K);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_gflops += calculateGFLOPS(M, N, K, ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    return total_gflops / runs;
}

// =========================================================================
// Correctness verification (compare kernel output vs cuBLAS)
// =========================================================================
namespace detail {

template <typename T>
bool verifyResult(const T *ref, const T *kernel, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        float ref_val, kernel_val;
        if constexpr (std::is_same_v<T, half>) {
            ref_val = __half2float(ref[i]);
            kernel_val = __half2float(kernel[i]);
        } else {
            ref_val = static_cast<float>(ref[i]);
            kernel_val = static_cast<float>(kernel[i]);
        }

        float abs_err = std::fabs(ref_val - kernel_val);
        float rel_err =
            std::fabs((ref_val - kernel_val) /
                      (ref_val != 0.0f ? ref_val : 1.0f));

        // FP32: 4096 accumulations can produce ~hundreds of ULPs difference
        // due to different FMA/reduction orders vs cuBLAS
        const float abs_tol = std::is_same_v<T, float> ? 500.0f : 10.0f;
        const float rel_tol = std::is_same_v<T, float> ? 0.03f : 0.05f;

        if (abs_err > abs_tol && rel_err > rel_tol) {
            std::fprintf(stderr,
                         "Mismatch[%d]: ref=%.6f kernel=%.6f "
                         "abs=%.6f rel=%.2f%%\n",
                         i, ref_val, kernel_val, abs_err, rel_err * 100);
            return false;
        }
    }
    return true;
}

} // namespace detail

template <typename T>
bool testCorrectness(int M, int N, int K,
                     typename LauncherTraits<T>::Launcher launcher) {
    if (launcher == nullptr) return true;

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(T);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(T);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(T);

    T *h_a = static_cast<T *>(std::malloc(a_bytes));
    T *h_b = static_cast<T *>(std::malloc(b_bytes));
    T *h_ref = static_cast<T *>(std::malloc(c_bytes));
    T *h_kernel = static_cast<T *>(std::malloc(c_bytes));
    initMatrix(h_a, M, K, 1.0f);
    initMatrix(h_b, K, N, 2.0f);

    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);
    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

    // Run cuBLAS for reference
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        double dummy;
        T *host_ref = nullptr;
        if constexpr (std::is_same_v<T, float>)
            detail::runCublasSgemm(M, N, K, d_a, d_b, d_c, handle, start, stop,
                                   dummy, host_ref);
        else
            detail::runCublasHgemm(M, N, K, d_a, d_b, d_c, handle, start, stop,
                                   dummy, host_ref);
        cudaMemcpy(h_ref, d_c, c_bytes, cudaMemcpyDeviceToHost);
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::free(host_ref);
    }

    // Run kernel
    cudaMemset(d_c, 0, c_bytes);
    launcher(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_kernel, d_c, c_bytes, cudaMemcpyDeviceToHost);

    bool ok = detail::verifyResult(h_ref, h_kernel, M, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_ref);
    std::free(h_kernel);
    return ok;
}

// =========================================================================
// Convenience: correctness + performance for one kernel
// =========================================================================
template <typename T>
void checkKernel(int M, int N, int K,
                 typename LauncherTraits<T>::Launcher launcher,
                 const char *label) {
    std::printf("  %-20s", label);
    if (!testCorrectness<T>(M, N, K, launcher)) {
        std::printf("FAIL (correctness)\n");
        return;
    }
    double gflops = testGEMM<T>(M, N, K, launcher);
    std::printf("✓  %.2f GFLOPS\n", gflops);
}
