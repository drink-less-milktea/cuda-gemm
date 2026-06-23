# 性能数据与分析

> H100 SXM, CUDA 12.6, 4096×4096 矩阵乘法

---

## 1. 总体性能对比

### SGEMM (FP32, CUDA Core)

| 版本 | 单次耗时 (ms) | GFLOPS | 加速比 | 相对 cuBLAS |
|------|:----------:|:------:|:------:|:----------:|
| naive | 35.19 | 3,913 | 1.0× | 9.8% |
| v1 Shared Mem Tiling | 5.63 | 24,543 | 6.3× | 61.6% |
| v2 Bank Conflict Avoid. | 4.86 | 28,190 | 7.2× | 70.7% |
| v3 Double Buffering | 3.64 | **37,898** | **9.7×** | **95.1%** |
| cuBLAS (cutlass) | ~3.46 | 39,868 | — | 100% |

### HGEMM (FP16, Tensor Core WMMA)

| 版本 | 单次耗时 (ms) | GFLOPS | TFLOPS | 加速比 | 相对 cuBLAS |
|------|:----------:|:------:|:------:|:------:|:----------:|
| v1 WMMA | 0.900 | 151,941 | 151.9 | 1.00× | 33.4% |
| v2 +cp.async.ca | 0.857 | 159,612 | 159.6 | 1.05× | 35.1% |
| v3 +Double Buffering | 0.641 | 212,803 | 212.8 | 1.40× | 46.7% |
| v4 +cp.async.cg+3D Grid | 0.704 | 194,258 | 194.3 | 1.28× | 42.7% |
| cuBLAS | ~0.278 | ~455,265 | 455.3 | — | 100% |

---

## 2. Nsight Compute 硬件指标分析

### HGEMM v3 (最佳版本)

| 指标 | 数值 | 含义 |
|------|:----:|------|
| **SM Throughput** | **45.2%** | Tensor Core 利用率——核心计算单元用了不到一半 |
| **DRAM Throughput** | 12.5% | 内存带宽几乎空闲 → **compute-bound** |
| **L1/TEX Throughput** | 50.4% | Shared memory / L1 是最大瓶颈 |
| **Warp Occupancy** | 12.5% | 每 SM 每周期仅 2 warps 活跃（理论最大 16） |
| Grid Dim | (16, 32, 1) | N 方向 16 blocks, M 方向 32 blocks |
| Block Dim | (256, 1, 1) | 8 warps/block |

### HGEMM v1 (基线)

| 指标 | 数值 |
|------|:----:|
| SM Throughput | 33.3% |
| DRAM Throughput | 9.1% |
| L1/TEX Throughput | 47.1% |

### naive SGEMM (对照)

| 指标 | 数值 |
|------|:----:|
| SM Throughput | 64.1% |
| DRAM Throughput | 7.0% |
| **L1/TEX Throughput** | **96.2%** |

---

## 3. 瓶颈分析

### 3.1 为什么 HGEMM v3 比 cuBLAS 差 ~2.1×？

**根本原因：Occupancy 太低（12.5%）。**

```
每个 SM 有 4 个 Tensor Core，每个 cycle 可执行 1 条 MMA 指令。
理论最大：16 warps/SM → 16 条 MMA/cycle
实际：      2 warps/SM  →  2 条 MMA/cycle → 仅 12.5% 利用率
```

**为什么 occupancy 这么低？**

1. **Grid 太小**：4096², BM=128, BN=256 → grid = 16×32 = **仅 512 blocks**。H100 有 132 SM，每 SM 分不到 4 blocks。
2. **Block 内 warp 太少**：256 threads = 8 warps，其中 2 个覆盖 M 维 (wid&1)，4 个覆盖 N 维 (wid>>1)。很多 warp 在等待同步。
3. **Shared memory 用量大**：双缓冲 ~54 KB/block，H100 每 SM 最大 228 KB → 最多 4 blocks/SM，但 grid 本身就小。

### 3.2 为什么 v3 比 v1 快 40%？

| 优化 | v1 | v3 | 机制 |
|------|:--:|:--:|------|
| 数据加载 | 同步 float4 | cp.async.ca + 双缓冲 | 计算与访存 overlap |
| Shared memory | 静态分配 27KB | 动态分配 54KB | 两倍空间换流水线 |
| SM Throughput | 33.3% | 45.2% | +35% Tensor Core 利用率 |

### 3.3 为什么 v4 反而比 v3 慢？

v4 加了 cp.async.cg（绕过 L1）和 3D Grid Split，但 4096² 矩阵下：
- **3D Grid Split 无效果**：4096 列 / 4096 NSPLIT = 1 split，grid 仍是 2D
- **cp.async.cg 引入额外开销**：绕过 L1 对 GEMM 场景没有收益（数据本就不会重用），cg 的 bypass 逻辑反而增加了 latency

**v4 的适用场景**：N > 16384 的超宽矩阵。

### 3.4 naive SGEMM 为什么 L1 利用率 96%？

naive kernel 直接从 global memory 读数据，每次读取经过 L1 cache。4096² 矩阵下 K=4096 次迭代，每线程读取 8192 floats = 32KB。L1 miss rate 100%，导致 L1/TEX 吞吐接近饱和。

---

## 4. 优化效果阶梯

```
SGEMM (FP32):
  naive:         3.91 GFLOPS  ▏
  v1 (smem):    24.43 GFLOPS  ████ (+6.2×)
  v2 (bank):    28.30 GFLOPS  █████ (+7.2×)
  v3 (dbl buf): 37.77 GFLOPS  ██████ (+9.7×)
  cuBLAS:       39.08 GFLOPS  ██████

HGEMM (FP16 Tensor Core):
  v1 (WMMA):         151.9 TFLOPS  ████
  v2 (+cp.async.ca):  159.6 TFLOPS  ████▏ (+5%)
  v3 (+dbl buf):      212.8 TFLOPS  █████▋ (+40%)
  v4 (+cg+3D):        194.3 TFLOPS  █████▎ (+28%)
  cuBLAS:             455.3 TFLOPS  ████████████
```

**关键洞察**：
- **SGEMM**：从 naive 到 v3 提升 9.7×，基本追平 cuBLAS。主要收益来自 shared memory tiling。
- **HGEMM**：v1 → v3 提升 40%，但离 cuBLAS 仍有 2.1× 差距。最大瓶颈是 **occupancy**（12.5%）。要实现进一步加速需要：
  - 增加 block 内 warp 数（更大的 tile 或 thread coarsening）
  - 使用 MMA PTX 替代 WMMA（更灵活的寄存器分配）
  - 探索 warp specialization（部分 warp 做 I/O，部分做 compute）

---

## 5. 不同矩阵大小的性能曲线

### HGEMM 各版本 vs 矩阵大小

| M=N=K | SGEMM v3 | vs cuBLAS | HGEMM v3 | vs cuBLAS |
|:-----:|:--------:|:---------:|:--------:|:---------:|
| 1024 | — | — | 52,910 GFLOPS (52.9 TFLOPS) | 74.8% |
| 4096 | **37,898 GFLOPS** | **95.1%** | 213,225 GFLOPS (213.2 TFLOPS) | 39.7% |
| 8192 | **36,799 GFLOPS** | **95.8%** | 243,061 GFLOPS (243.1 TFLOPS) | 40.5% |

> SGEMM v3 在 4096–8192 区间稳定在 cuBLAS 的 **95%+**，与原始 H100 测试图一致。

---

## 6. 方法论：如何复现

### 基本性能测试

```bash
cd build
./cuda_gemm 4096    # 快速验证（带正确性检查）
./cuda_gemm --bench  # 完整 benchmark（64 尺寸，输出 CSV）
```

### Nsight Systems 时间线分析

```bash
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:$PATH
nsys profile --stats=true -o gemm_profile ./cuda_gemm 4096
# 生成 gemm_profile.nsys-rep 和 gemm_profile.sqlite
```

### Nsight Compute 硬件指标

```bash
# 单 kernel 全量指标
ncu --set full --kernel-name v3Hgemm -o v3_ncu ./cuda_gemm 4096

# 快速关键指标
ncu --kernel-name v3Hgemm \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    ./cuda_gemm 4096
```
