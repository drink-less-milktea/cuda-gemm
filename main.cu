#include "common/benchmark.cuh"
#include "sgemm/naive_sgemm.cuh"
#include "sgemm/v1_sgemm.cuh"
#include "sgemm/v2_sgemm.cuh"
#include "sgemm/v3_sgemm.cuh"
#include "hgemm/v1_hgemm.cuh"
#include "hgemm/v2_hgemm.cuh"
#include "hgemm/v3_hgemm.cuh"
#include "hgemm/v4_hgemm.cuh"

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

// All square matrix sizes to test (256 → 16384, step 256)
static const int Ns[] = {
    256,   512,   768,   1024,  1280,  1536,  1792,  2048,  2304,  2560,
    2816,  3072,  3328,  3584,  3840,  4096,  4352,  4608,  4864,  5120,
    5376,  5632,  5888,  6144,  6400,  6656,  6912,  7168,  7424,  7680,
    7936,  8192,  8448,  8704,  8960,  9216,  9472,  9728,  9984,  10240,
    10496, 10752, 11008, 11264, 11520, 11776, 12032, 12288, 12544, 12800,
    13056, 13312, 13568, 13824, 14080, 14336, 14592, 14848, 15104, 15360,
    15616, 15872, 16128, 16384};
static constexpr int kNumSizes = sizeof(Ns) / sizeof(Ns[0]);

// =========================================================================
// Quick check: correctness + performance for a single size (used by main)
// =========================================================================
template <typename T>
void runCheck(const char *title, std::vector<typename LauncherTraits<T>::Launcher> launchers,
              const std::vector<const char *> &names, int M, int N, int K) {
    std::printf("\n=== %s (M=%d N=%d K=%d) ===\n", title, M, N, K);

    double cublas_gflops = testGEMM<T>(M, N, K, nullptr);
    std::printf("  cuBLAS            ✓  %.2f GFLOPS\n", cublas_gflops);

    for (size_t i = 0; i < launchers.size(); ++i) {
        checkKernel<T>(M, N, K, launchers[i], names[i]);
    }
}

// =========================================================================
// Full benchmark: all sizes, CSV output
// =========================================================================
template <typename T>
void runBenchmark(const char *csv_path,
                  std::vector<typename LauncherTraits<T>::Launcher> launchers,
                  const std::vector<const char *> &names) {
    std::ofstream ofs(csv_path);
    ofs << std::setprecision(6);
    ofs << "M";
    for (auto n : names) ofs << "," << n;
    ofs << "\n";

    for (int i = 0; i < kNumSizes; ++i) {
        int m = Ns[i];
        std::printf("  %s %d x %d...", LauncherTraits<T>::name, m, m);
        std::fflush(stdout);
        ofs << m;
        for (size_t j = 0; j < launchers.size(); ++j) {
            double gflops = testGEMM<T>(m, m, m, launchers[j]);
            ofs << "," << gflops;
        }
        ofs << "\n";
        std::printf(" done\n");
    }
    ofs.close();
    std::printf("  Results written to %s\n", csv_path);
}

// =========================================================================
int main(int argc, char **argv) {
    // Default: quick check at 4096
    int M = 4096;
    bool full_bench = false;

    if (argc > 1) {
        std::string arg(argv[1]);
        if (arg == "--bench") {
            full_bench = true;
        } else {
            M = std::stoi(arg);
        }
    }

    // ---- SGEMM ----
    {
        using L = LauncherTraits<float>::Launcher;
        std::vector<L> launchers = {naiveSgemmLauncher, v1SgemmLauncher,
                                    v2SgemmLauncher, v3SgemmLauncher};
        std::vector<const char *> names = {"naive_sgemm", "v1_sgemm",
                                           "v2_sgemm", "v3_sgemm"};

        if (full_bench) {
            runBenchmark<float>("sgemm_result.csv", launchers, names);
        } else {
            runCheck<float>("SGEMM", launchers, names, M, M, M);
        }
    }

    // ---- HGEMM ----
    {
        using L = LauncherTraits<half>::Launcher;
        std::vector<L> launchers = {v1HgemmLauncher, v2HgemmLauncher,
                                    v3HgemmLauncher, v4HgemmLauncher};
        std::vector<const char *> names = {"v1_hgemm", "v2_hgemm", "v3_hgemm",
                                           "v4_hgemm"};

        if (full_bench) {
            runBenchmark<half>("hgemm_result.csv", launchers, names);
        } else {
            runCheck<half>("HGEMM", launchers, names, M, M, M);
        }
    }

    std::printf("\nAll tests complete.\n");
    return 0;
}
