# CUDA GEMM

[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-H100-blue)](https://www.nvidia.com/en-us/data-center/h100/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 从零手写 CUDA GEMM kernel，逐步优化到接近 cuBLAS 性能。
> 8 个渐进版本，覆盖 FP32 CUDA Core 和 FP16 Tensor Core WMMA 两条优化线。

---

## 亮点概括

| 指标 | 数据 |
|------|------|
| **SGEMM v3** (FP32) | **37,898 GFLOPS @ 4096² — cuBLAS 的 95.1%** |
| **SGEMM v3** (FP32) | **36,799 GFLOPS @ 8192² — cuBLAS 的 95.8%** |
| **HGEMM v3** (FP16 Tensor Core) | **213 TFLOPS @ 4096², 243 TFLOPS @ 8192²** |
| **SGEMM 加速比** | naive → v3: **9.7×** |
| **HGEMM 加速比** | v1 → v3: **1.40×** |
| **覆盖技术** | Shared Mem Tiling, Bank Conflict Avoidance, Double Buffering, WMMA, cp.async, 3D Grid Split |
| **硬件指标** | Nsight Compute: SM throughput 45%, DRAM 12.5% (compute-bound), L1/TEX 50% (bottleneck) |
| **测试框架** | 正确性验证 (vs cuBLAS) + 64 尺寸 benchmark + CSV 输出 |
| **文档** | [`OPTIMIZATION.md`](docs/OPTIMIZATION.md) 优化原理详解 + [`PERFORMANCE.md`](docs/PERFORMANCE.md) profiling 数据 |

---

## 快速开始

### 环境要求

- NVIDIA GPU (SM80+, Ampere 及以上)
- CUDA Toolkit 12.x
- CMake ≥ 3.21

### 构建 & 运行

```bash
# 构建
mkdir build && cd build
cmake .. && make -j$(nproc)

# 快速验证（4096×4096，含正确性检查）
./cuda_gemm

# 指定矩阵大小
./cuda_gemm 2048

# 完整 benchmark（64 个尺寸，输出 CSV）
./cuda_gemm --bench
```

---

## 优化路线图

### SGEMM — FP32 CUDA Core

```
naive ──→ v1: Shared Memory Tiling + Float4 + Thread Coarsening
                ↓
           v2: Register Buffering + Bank Conflict Avoidance
                ↓
           v3: Double Buffering (Software Pipelining)
```

### HGEMM — FP16 Tensor Core (WMMA)

```
 v1: WMMA Tile GEMM ──→ v2: + cp.async.ca (异步拷贝)
                              ↓
                         v3: + Double Buffering + Dynamic SMEM
                              ↓
                         v4: + cp.async.cg + 3D Grid Split
```

---

## 项目结构

```
cuda-gemm/
├── main.cu                       # 统一入口
├── CMakeLists.txt                # CMake 构建
├── common/
│   ├── utils.cuh                 # 公共工具
│   └── benchmark.cuh             # 模板 benchmark 框架
├── sgemm/                        # FP32 GEMM kernels
│   ├── naive_sgemm.cu[h]
│   ├── v1_sgemm.cu[h]            # Shared memory tiling
│   ├── v2_sgemm.cu[h]            # Bank conflict avoidance
│   └── v3_sgemm.cu[h]            # Double buffering
├── hgemm/                        # FP16 Tensor Core GEMM kernels
│   ├── v1_hgemm.cu[h]            # WMMA baseline
│   ├── v2_hgemm.cu[h]            # + cp.async.ca
│   ├── v3_hgemm.cu[h]            # + double buffering + dynamic smem
│   └── v4_hgemm.cu[h]            # + cp.async.cg + 3D grid split
└── docs/
    ├── OPTIMIZATION.md            # 面试讲稿级详解
    └── PERFORMANCE.md            # 性能数据 + profiling
```

---

## 关键技术索引

| 技术 | 使用版本 |
|------|---------|
| Shared Memory Tiling | SGEMM v1,v2,v3 / HGEMM 全版本 |
| Float4 向量化 | SGEMM v1,v2,v3 / HGEMM v1 |
| Thread Coarsening (TM×TN) | SGEMM v1,v2,v3 |
| Bank Conflict Avoidance | SGEMM v2 |
| Register Buffering | SGEMM v2,v3 |
| Double Buffering | SGEMM v3 / HGEMM v3,v4 |
| Dynamic Shared Memory | HGEMM v3,v4 |
| WMMA (Tensor Core) | HGEMM 全版本 |
| cp.async (异步拷贝) | HGEMM v2,v3,v4 |
| cp.async.cg (Cache Bypass) | HGEMM v4 |
| 3D Grid Split | HGEMM v4 |

---

## 已知限制

- HGEMM v4 在 N < 4096 的矩阵上 3D Grid Split 不生效，性能略低于 v3（预期行为）
- 所有 kernel 要求 M,N,K 对齐 BM/BN/BK 块大小（SGEMM: 128/128/8, HGEMM: 128/256/32）

## 参考文献

- [CUDA C++ Programming Guide — Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA 的模板 GEMM 库
- [How to Optimize a CUDA GEMM Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm 的经典教程

## License

MIT © 2025 Cai Yiwen — 详见 [LICENSE](LICENSE)
