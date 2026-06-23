# CUDA GEMM

[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-H100-blue)](https://www.nvidia.com/en-us/data-center/h100/)

> 从零手写 CUDA GEMM kernel，逐步优化到接近 cuBLAS 性能。
> 涵盖 FP32 (CUDA Core) 和 FP16 (Tensor Core WMMA) 两条优化线，
> 共 8 个渐进版本，展示 GPU 架构优化的完整技能栈。

---

## 项目亮点

- **8 个 kernel 版本**：从 naive global memory 到 Tensor Core + 异步拷贝 + 双缓冲 + 3D grid
- **完整 benchmark 框架**：正确性验证（vs cuBLAS）+ 性能测试 + CSV 输出
- **H100 实测**：在 NVIDIA H100 上测试，FP16 Tensor Core 最高 ~53 TFLOPS (1024²)
- **面试导向文档**：[`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — 面试讲稿级详解

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
├── main.cu                       # 统一入口（快速验证 / 完整 benchmark）
├── CMakeLists.txt                # CMake 构建
├── common/
│   ├── utils.cuh                 # 公共工具（OFFSET/FLOAT4/GFLOPS/矩阵初始化）
│   └── benchmark.cuh             # 模板 benchmark 框架（正确性 + 性能）
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
├── scripts/
│   ├── run_all.sh                # 一键 benchmark
│   └── plot.py                   # 性能绘图
└── docs/
    ├── OPTIMIZATION.md            # 面试讲稿级详解
    └── PERFORMANCE.md            # 性能数据和分析
```

---

## 关键技术索引

| 技术 | 使用版本 | 面试价值 |
|------|---------|---------|
| Shared Memory Tiling | SGEMM v1,v2,v3 / HGEMM 全版本 | ⭐⭐⭐⭐⭐ |
| Float4 向量化 | SGEMM v1,v2,v3 / HGEMM v1 | ⭐⭐⭐⭐ |
| Thread Coarsening (TM×TN) | SGEMM v1,v2,v3 | ⭐⭐⭐⭐ |
| Bank Conflict Avoidance | SGEMM v2 | ⭐⭐⭐⭐ |
| Register Buffering | SGEMM v2,v3 | ⭐⭐⭐ |
| Double Buffering | SGEMM v3 / HGEMM v3,v4 | ⭐⭐⭐⭐⭐ |
| Dynamic Shared Memory | HGEMM v3,v4 | ⭐⭐⭐ |
| WMMA (Tensor Core) | HGEMM 全版本 | ⭐⭐⭐⭐⭐ |
| cp.async (异步拷贝) | HGEMM v2,v3,v4 | ⭐⭐⭐⭐⭐ |
| cp.async.cg (Cache Bypass) | HGEMM v4 | ⭐⭐⭐ |
| 3D Grid Split | HGEMM v4 | ⭐⭐⭐ |

---

## 已知问题

- **SGEMM 正确性**：FP32 kernel 与 cuBLAS 存在约 1-5% 的数值误差。原因是原始实现未考虑 K 维度不对齐和浮点舍入顺序差异。HGEMM (Tensor Core) 版本在 ≥1024 尺寸下验证通过。（详见 [#known-issues])
- SGEMM 和 HGEMM 的 small-size 正确性（<512）需要进一步调试。

---

## 参考文献

- [CUDA C++ Programming Guide — Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA 的模板 GEMM 库
- [How to Optimize a CUDA GEMM Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm 的经典教程
