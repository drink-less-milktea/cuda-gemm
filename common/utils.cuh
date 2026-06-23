#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda_device_runtime_api.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

/// Row-major 2D offset macro.
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/// Reinterpret a scalar as a float4 for 128-bit vectorised loads/stores.
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

/// Fill a float matrix with deterministic pseudo-random values.
inline void initMatrix(float *mat, int rows, int cols, float value = 1.0f) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = value + static_cast<float>(i % 100) * 0.01f;
}

/// Fill a half matrix with deterministic pseudo-random values.
inline void initMatrix(half *mat, int rows, int cols, float value = 1.0f) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = __float2half(value + static_cast<float>(i % 100) * 0.01f);
}

// ---------------------------------------------------------------------------
// Performance helpers
// ---------------------------------------------------------------------------

/// Compute GFLOPS: 2*M*N*K multiply-adds / time_seconds / 1e9.
inline double calculateGFLOPS(int M, int N, int K, double time_seconds) {
    double ops = 2.0 * static_cast<double>(M) * N * K;
    return ops / (time_seconds * 1e9);
}

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

inline void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg,
                     cudaGetErrorString(result));
        std::exit(1);
    }
}

inline void checkCublas(cublasStatus_t result, const char *msg) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error %s: %d\n", msg,
                     static_cast<int>(result));
        std::exit(1);
    }
}
