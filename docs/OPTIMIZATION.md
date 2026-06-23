# CUDA GEMM 优化历程：从 Naive 到 Tensor Core

> 面试导向的技术文档——每个优化步骤的**原理 → 实现 → 效果**。
> 适合面试前复习，每个小节都可以展开为 5-10 分钟的深度讨论。

---

## 目录

1. [背景：为什么手写 GEMM](#1-背景为什么手写-gemm)
2. [SGEMM 优化线（FP32, CUDA Core）](#2-sgemm-优化线fp32-cuda-core)
   - [v0 Naive: 基线实现](#v0-naive-基线实现)
   - [v1 Shared Memory Tiling: 突破内存墙](#v1-shared-memory-tiling-突破内存墙)
   - [v2 Bank Conflict Avoidance: 消除 shared memory 瓶颈](#v2-bank-conflict-avoidance-消除-shared-memory-瓶颈)
   - [v3 Double Buffering: 计算与访存 overlap](#v3-double-buffering-计算与访存-overlap)
3. [HGEMM 优化线（FP16, Tensor Core WMMA）](#3-hgemm-优化线fp16-tensor-core-wmma)
   - [v1 WMMA Tile GEMM: Tensor Core 入门](#v1-wmma-tile-gemm-tensor-core-入门)
   - [v2 cp.async: 异步拷贝替代同步加载](#v2-cpasync-异步拷贝替代同步加载)
   - [v3 Double Buffering + Dynamic SMEM: 软件流水线](#v3-double-buffering--dynamic-smem-软件流水线)
   - [v4 cp.async.cg + 3D Grid Split: Cache 控制与超宽矩阵](#v4-cpasynccg--3d-grid-split-cache-控制与超宽矩阵)
4. [面试高频问题速答](#4-面试高频问题速答)
5. [性能总结](#5-性能总结)

---

## 1. 背景：为什么手写 GEMM

### 面试开场白

> "矩阵乘法（GEMM）是深度学习推理/训练的核心算子，占据了 Transformer 中 50%+ 的计算量。
> 我通过从零手写 CUDA GEMM kernel，逐步优化到接近 cuBLAS 的性能，深入理解了 GPU 架构的各个层次：
> 全局内存合并访问、共享内存 bank conflict、warp 调度、Tensor Core 编程模型、异步拷贝流水线等。"

### 关键数字（H100 SXM, CUDA 12.6）

| 指标 | 数值 |
|------|------|
| FP32 理论峰值 | 67 TFLOPS |
| FP16 Tensor Core 峰值 | 990 TFLOPS |
| HBM3 带宽 | 3.35 TB/s |
| L2 Cache | 50 MB |
| SM 数量 | 132 |
| Max shared memory / SM | 228 KB (可配) |
| Max shared memory / block | 228 KB (opt-in) |

### 为什么手写能接近甚至超过 cuBLAS？

cuBLAS 是通用库，需要处理任意大小、任意 leading dimension、任意 transpose 组合。手写 kernel 可以：
- 针对特定 tile size 做死循环展开
- 用编译期常量消除运行时分支
- 精确控制寄存器分配和 shared memory 布局

---

## 2. SGEMM 优化线（FP32, CUDA Core）

### 通用参数设定

| 参数 | 值 | 理由 |
|------|-----|------|
| BM (block tile M) | 128 | 平衡 shared memory 用量与并行度 |
| BN (block tile N) | 128 | 同上 |
| BK (block tile K) | 8 | FP32 需要 4B/元素，BK 太大会导致 shared memory 溢出 |
| TM × TN | 8 × 8 | 每个线程计算 64 个输出元素 (thread coarsening) |
| 线程数/block | 256 (16×16) | 8 warps，平衡 occupancy 与资源 |

---

### v0 Naive: 基线实现

**原理**：每个线程负责一个输出元素 `C[m][n]`，直接遍历整个 K 维。

```cuda
// 核心循环：每线程 1 个结果
float sum = 0;
for (int k = 0; k < K; k++)
    sum += A[m * K + k] * B[k * N + n];
C[m * N + n] = sum;
```

**性能瓶颈分析**：

| 瓶颈 | 数值 | 说明 |
|------|------|------|
| Global memory 访问量 | 每个输出元素读取 2K 次 global memory | K=1024 时每个线程读 8192 bytes |
| 计算访存比 | O(1) FLOP/byte | 内存带宽完全饱和 |
| 预期性能 | < 5% 峰值 | 完全受限于 HBM 带宽 |

**面试要点**：
- "这是一个 memory-bound kernel，每个 FMA 需要两次 global memory load"
- "没有数据复用——A 的每一行被 N 个线程重复读取 N 次，B 的每一列被 M 个线程重复读取 M 次"

---

### v1 Shared Memory Tiling: 突破内存墙

**核心思想**：将矩阵分块（tile），每个 block 把当前 tile 加载到 shared memory，让 block 内所有线程共享。

```
┌──────────────────┐
│  A (M×K)         │    ┌──────┐
│  ┌──────┐        │    │128×8 │  ← A tile (BM×BK)
│  │128×8 │ ...    │    └──────┘
│  └──────┘        │
│  ...     ...     │    ┌──────┐
│                  │    │ 8×128│  ← B tile (BK×BN)
│  B (K×N)         │    └──────┘
│  ┌──────┐        │
│  │ 8×128│ ...    │
│  └──────┘        │
└──────────────────┘
```

**关键优化**：

1. **Float4 向量化加载**：`FLOAT4(s_a[...]) = FLOAT4(a[...])` 一次 load 128 bits (4 floats)，充分利用内存带宽
2. **Thread coarsening (TM=TN=8)**：每个线程算 8×8=64 个结果，减少线程数和调度开销
3. **数据复用比**：A tile 被 BN/TN=16 个线程复用，B tile 被 BM/TM=16 个线程复用

**代码关键段**：

```cuda
// 128 个线程合作加载 128×8 的 A tile
int load_a_smem_m = tid >> 1;      // 行：0..127
int load_a_smem_k = (tid & 1) << 2; // 列：0, 4（float4 = 4 floats）

// 每个线程加载一行中的 4 个连续元素
FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) =
    FLOAT4(a[OFFSET(load_gmem_m, load_gmem_k, K)]);
```

**面试要点**：
- "Shared memory 延迟约 20 cycles vs global memory 约 300 cycles——这是性能提升的根本原因"
- "Thread coarsening: 一个线程算 64 个结果，减少线程数，提高每个线程的计算密度"
- "Float4 向量化确保 128-bit 合并访问，单次 transaction 传输 32 bytes"

**预期效果**：从 baseline 的 ~1% 峰值提升到 ~10-20%

---

### v2 Bank Conflict Avoidance: 消除 shared memory 瓶颈

**问题**：v1 中 A 矩阵在 shared memory 里按 `s_a[BM][BK]`（128×8）行主序存储。当 warp 内多个线程同时访问同一 bank 的不同地址时，会发生 **bank conflict**。

**Shared Memory Bank 原理**（Volta+）：
- 32 个 bank，每个 bank 4 bytes 宽
- 同一 cycle 内，如果 ≥2 个线程访问同一 bank 的不同地址 → bank conflict → 串行化
- 同一 bank 的相同地址 → broadcast（无冲突）

**v1 的 bank conflict 场景**：
```cuda
// 线程 tx 读取 s_a[ty*8 + m][k]
// 当不同 ty 的线程访问 s_a[不同行][同一 k] 时
// 步长 = 8 floats × 4 bytes = 32 bytes = 8 banks
// stride=8 → 32/8=4-way bank conflict
s_a[ty*8 + m][k]  // ty=0→bank0, ty=4→bank0 (32-byte stride)
```

**解决方案**：A 矩阵在 shared memory 中转置存储 `s_a[BK][BM]`（8×128）。

```cuda
// v2: A 转置存储 → 无 bank conflict
__shared__ float s_a[BK][BM];  // 8×128，访问步长 = 128 floats

// 加载时转置：
s_a[k][m] = r_load_a[...];  // 列主序存储

// 计算时读取：
FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty*TM/2]);
// stride = BM/2 = 64 floats × 4B = 256B → bank stride = 64
// 所有线程访问同一 bank 的不同行 → 所有在同一 bank → CONFLICT!
```

等等，这仍然会有 bank conflict。实际上，在原始代码中，v2 的真正改进是：
- 先加载到寄存器（`r_load_a[4]`），再写入 shared memory——这样可以减少 shared memory 的 bank conflict，因为写入比读取冲突少
- B 矩阵保持不变
- 计算时将 shared memory 中的数据再次加载到寄存器（`r_comp_a[8], r_comp_b[8]`），后续在寄存器上做全部计算

**实际改进机制**：

v1 的瓶颈在内层循环：
```cuda
for (int k = 0; k < BK; k++)
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            r_c[m][n] += s_a[ty*TM + m][k] * s_b[k][tx*TN + n];
```

这里每轮都要从 shared memory 读 A 和 B 各 64 次（TM×TN 次乘加）。BK=8 → 8×64×2 = 1024 次 shared memory 读取。

v2 将数据先读到寄存器：
```cuda
// 每 BK 步：先 load
FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty*TM/2]);
FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty*TM/2 + BM/2]);
// 再计算（全寄存器操作，零 shared memory 访问）
for (int tm = 0; tm < TM; tm++)
    for (int tn = 0; tn < TN; tn++)
        r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
```

这一步将 shared memory 读取从 **1024 次** 降低到 **(8+8)/4=4 次 float4 load**（A 和 B 各 4 个 float 的向量化加载）。

**面试要点**：
- "寄存器→寄存器运算没有 bank conflict，延迟最低（~0 cycles）"
- "Shared memory 带宽约 128 bytes/cycle/SM，避免 bank conflict 才能充分利用"
- "Vectorized load (float4) + register buffering 是组合拳"
- "A 转置存 shared memory 是为了写入无冲突；读取时通过寄存器重排避免冲突"

---

### v3 Double Buffering: 计算与访存 overlap

**问题**：v1/v2 中，每个 tile 的计算和加载是**串行**的：

```
[Load A₀,B₀] → [sync] → [Compute₀] → [sync] → [Load A₁,B₁] → ...
```

加载时 GPU 计算单元空闲，计算时内存单元空闲——互相等待。

**解决方案**：双缓冲（double buffering / software pipelining）。

```
Buffer 0: [Load A₀,B₀] → [     Compute₀     ] → ...
Buffer 1:                 [Load A₁,B₁]                  → [Compute₁] → ...
```

**实现**：
```cuda
__shared__ float s_a[2][BK][BM];  // 两份缓冲
__shared__ float s_b[2][BK][BN];

// 预加载 buffer 0
{ load buffer 0 from global; __syncthreads(); }

for (int bk = 1; bk < K/BK; bk++) {
    int cur = (bk-1) & 1;   // 当前计算缓冲
    int next = bk & 1;       // 下一个加载缓冲

    // 发起下一块的数据加载（非阻塞）
    FLOAT4(r_load_a) = FLOAT4(a[next_addr]);

    // 同时用当前缓冲计算
    for (int tk = 0; tk < BK; tk++) {
        load_from_smem(s_a[cur]);  // 从当前缓冲读
        compute_on_registers();
    }

    // 将加载的数据写入下一个缓冲
    s_a[next][k][m] = r_load_a;
    __syncthreads();
}
```

**关键约束**：
- 需要 2× shared memory（从 ~32KB 增加到 ~64KB for SGEMM）
- 计算量必须足够大以隐藏访存延迟
- BM×BK 和 BK×BN tile 需要合理设计

**面试要点**：
- "软件流水线的本质是用空间换时间——两倍 shared memory 换来计算和访存的重叠"
- "GPU 有独立的 load/store 单元和计算单元，double buffering 让两者同时工作"
- "需要权衡：shared memory 翻倍会降低 occupancy，需要确保计算量足够覆盖延迟"

---

## 3. HGEMM 优化线（FP16, Tensor Core WMMA）

### Tensor Core 基础

- **硬件单元**：每个 SM 有 4 个 Tensor Core（Volta/Turing），Ampere+ 翻倍
- **WMMA API**：`wmma::mma_sync` 在 warp 级别调用 Tensor Core
- **MMA 形状**：`m16n16k16` — 每个 warp 每步做 16×16×16=4096 次乘加
- **吞吐**：H100 上 FP16 Tensor Core 峰值 990 TFLOPS vs FP32 CUDA Core 67 TFLOPS — **~15× 差距**

### 通用参数设定

| 参数 | 值 | 理由 |
|------|-----|------|
| BM × BN × BK | 128 × 256 × 32 | BK=32 是 WMMA k16 的 2 倍，支持 2-stage k 展开 |
| Warp 数量 | 8 (256 threads) | 2 warps 覆盖 M 维，4 warps 覆盖 N 维 |
| WMMA 瓦片数/warp | 2 k-stage × 4 tiles × 4×4 accum = 16 个 MMA 操作/warp/main loop |

---

### v1 WMMA Tile GEMM: Tensor Core 入门

**核心架构**：
```
Block 负责: C[128×256] tile
  8 warps: 2×4 瓦片布局
    每个 warp 负责: 64×64 输出
      4×4 = 16 个 WMMA m16n16k16 操作
        每个操作 = 4096 FMA (Tensor Core)
```

**Shared memory 布局**：
```cuda
__shared__ half s_a[BM][BK + APAD];  // 128 × (32+8) = 128×40
__shared__ half s_b[BK][BN + BPAD];  //  32 × (256+8) = 32×264
```

**Padding (APAD=8, BPAD=8)**：在每行末尾加 8 个元素 padding，使得相邻行的相同列偏移至少相隔 (BK+8)×2=80 bytes，分散到不同 bank，避免 bank conflict。

**加载策略**：
```cuda
// 256 个线程，每个加载 float4 = 8 个 half
// A: 每个线程加载 2 行（load_a_smem_m 和 +1）
// B: 每个线程加载 4 行（load_b_smem_k 到 +3）
FLOAT4(s_a[m][k]) = FLOAT4(a[addr]);      // 8 halfs/thread
FLOAT4(s_b[0..3][n]) = FLOAT4(b[addr*0..3]); // 32 halfs/thread
```

**面试要点**：
- "Tensor Core 是专用硬件，mma_sync 在 1 个 cycle 内完成 16×16×16 FMA——相当于 4096 次 FP32 运算"
- "WMMA API 屏蔽了寄存器文件重排的复杂性，但限制了灵活性"
- "Padding 技巧在很多高性能 kernel 中都会用到"

---

### v2 cp.async: 异步拷贝替代同步加载

**问题**：v1 用 `FLOAT4(...) = FLOAT4(a[...])` 做同步 global→shared 拷贝。线程在拷贝完成前阻塞。

**SM80+ 新指令**：`cp.async` (asynchronous copy) — 从 Ampere (SM80) 开始支持。

```cuda
// v1: 同步加载（线程阻塞等待完成）
FLOAT4(s_a[m][k]) = FLOAT4(a[addr]);

// v2: 异步加载 + 显式同步
asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
    "r"(smem_addr), "l"(&a[gmem_addr]));
// ... 发起更多 cp.async ...
asm("cp.async.commit_group;\n" ::);   // 提交异步拷贝组
asm("cp.async.wait_group 0;\n" ::);   // 等待所有拷贝完成
```

**cp.async 的优势**：
1. **非阻塞**：线程发起拷贝后可以继续执行（不过 v2 暂时没有 overlap，v3 才有）
2. **L1 cache bypass**：`ca` = cache-all，数据直接进 shared memory，不污染 L1
3. **硬件 pipeline**：cp.async 有自己的硬件队列，不与 load/store 单元竞争

**面试要点**：
- "cp.async 是 PTX 指令，比内联 PTX 比编译器生成的 load 指令更高效"
- "ca 后缀表示 cache-all——数据绕过 L1 直接写入 shared memory，避免不必要的数据搬移"
- "commit_group + wait_group 提供了比 __syncthreads 更细粒度的同步控制"

---

### v3 Double Buffering + Dynamic SMEM: 软件流水线

**核心创新**：将 SGEMM v3 的双缓冲思想带到 Tensor Core 场景。

**挑战**：HGEMM tile 更大（BM×BN=128×256 vs 128×128），双缓冲的内存开销更大。

```cuda
// 动态 shared memory：在 kernel launch 时指定大小
extern __shared__ half smem[];
half *s_a = smem;                              // Buffer 0 A tile
half *s_b = smem + 2 * BM * (BK + APAD);       // Buffer 0 B tile
// Buffer 1 紧接其后
int s_a_db_offset = BM * (BK + APAD);          // A buffer stride
int s_b_db_offset = BK * (BN + BPAD);          // B buffer stride
```

**内存计算**：
- 单 buffer：128×40 + 32×264 = 5120 + 8448 = 13,568 halfs ≈ 27 KB
- 双 buffer：≈ 54 KB
- 需要 `cudaFuncSetAttribute(kernel, MaxDynamicSharedMemorySize, 98304)` opt-in

**流水线设计**：
```
bk=0: [Load Buf0] → [wait] → [sync]
bk=1:               [Load Buf1] → [Compute Buf0] → [wait] → [sync]
bk=2:                              [Load Buf0] → [Compute Buf1] → [wait] → [sync]
...
Final:                                             [Compute LastBuf]
```

**面试要点**：
- "动态 shared memory 允许 runtime 决定分配量，vs 静态 `__shared__` 在编译期固定"
- "这个优化在 H100 上效果显著，因为 H100 有更大的 shared memory (228KB/SM)"
- "双缓冲的索引翻转 `(bk & 1) ^ 1` 是一个常见的 ping-pong 技巧"

---

### v4 cp.async.cg + 3D Grid Split: Cache 控制与超宽矩阵

**两个独立优化**：

#### 4.1 cp.async.cg — Cache 策略

```cuda
// v2/v3: ca = cache-all (数据进 L1 + shared memory)
asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ...);
// v4:    cg = cache-global (绕过 L1，仅进 shared memory)
asm("cp.async.cg.shared.global [%0], [%1], 16;\n" ...);
```

- **ca** (cache at all levels): 数据同时进 L1 cache 和 shared memory。GEMM 场景下次不会再读同样的 global memory 地址，L1 缓存是浪费。
- **cg** (cache at global level): 绕过 L1，只写 shared memory。节省 L1 带宽给其他数据。

#### 4.2 3D Grid Split

**问题**：当 N 很大（如 16384），BN=256 时 `gridDim.x = 16384/256 = 64`，只有 64 个 block。H100 有 132 SM，大量 SM 空闲。

**解决**：用 gridDim.z 切分 x 维度，让更多 block 并发运行。

```cuda
// v1-v3: 2D grid
dim3 grid(BX, BY);
int bx = blockIdx.x;

// v4: 3D grid，x 维度被切到 z 维度
int split_num = (N + NSPLIT - 1) / NSPLIT;  // 按 4096 列切分
dim3 grid((BX + split - 1) / split, BY, split);
int bx = blockIdx.z * gridDim.x + blockIdx.x;  // 展平

// Boundary guard（因为增加了 grid 维度后会有多余 block）
if (bx >= BX || by >= BY) return;
```

**效果**：
- 宽矩阵（N=16384）：gridDim.x 从 64 → (64/split) × split 个 block
- SM 利用率从 64/132=48% 提升到接近 100%

**面试要点**：
- "Cache 策略是 GPU 性能调优的高级话题——理解何时 bypass L1 是关键"
- "3D grid 是一种网格拉伸技术，增加了 block 级并行度"
- "boundary guard `if (bx >= N/BN) return` 是一种低开销的 early exit"

---

## 4. 面试高频问题速答

### Q: 为什么 HGEMM 比 SGEMM 快这么多？（70K vs 23K GFLOPS）

- Tensor Core 的硬件吞吐远超 CUDA Core（~15× 差距）
- FP16 数据量是 FP32 的一半，内存带宽压力减半
- WMMA m16n16k16 每个 warp 每 cycle 算 4096 FMA，远超 scalar FMA

### Q: Shared memory bank conflict 怎么检测和解决？

- **检测**：nvprof/ncu 的 `shared_load_bank_conflict` / `shared_store_bank_conflict` 计数器
- **解决**：padding 改变列步长、数据转置存储、使用 `__cvta_generic_to_shared` 检查地址

### Q: Occupancy 和性能不一定正相关，为什么？

- 高 occupancy 意味着更多 warp 可调度，但每个 warp 的寄存器/shared memory 减少
- 本项目中，我们选择低 occupancy（256 threads/block, 54KB smem），专注于每个 warp 的计算密度
- 关键指标是 **latency hiding**：只要 warp 数量足够隐藏访存延迟即可

### Q: 为什么 BK 选择 8/32 而不是更大？

- BK 越大，tile 越大，shared memory 需求平方增长
- SGEMM BK=8：128×8=1K floats=4KB → 双缓冲 8KB。如果用 BK=16 会翻倍
- HGEMM BK=32：匹配 WMMA k=16 的 2 倍——恰好能做 2-stage k 展开
- BK 的选择是 shared memory size、寄存器压力、k 维度展开粒度的 trade-off

### Q: WMMA vs MMA (PTX) 的区别？

- WMMA 是 CUDA C++ API，MMA 是 inline PTX (`asm("mma.sync.aligned...");`)
- WMMA 更高级更安全，编译器处理寄存器分配
- MMA 更灵活：可以指定 `m16n8k16` 等非对称形状，可以处理 sparse、B1 等
- 本项目的 WMMA 版本适合快速原型和教学

---

## 5. 性能总结

### HGEMM (FP16 Tensor Core) — H100

| 版本 | 关键技术 | 4096×4096 性能 | vs v1 |
|------|----------|:---:|:---:|
| v1 | WMMA + Shared Memory | baseline | 1.0× |
| v2 | + cp.async.ca | ~+8% | 1.08× |
| v3 | + Double Buffering | ~+18% | 1.18× |
| v4 | + cp.async.cg + 3D Grid | ~+17% | 1.17× |

### SGEMM (FP32 CUDA Core) — H100

| 版本 | 关键技术 | 效果 |
|------|----------|------|
| naive | 直接 global memory | baseline |
| v1 | Shared Memory Tiling + Float4 + Thread Coarsening | ~5-10× |
| v2 | + Register Buffering + Bank Conflict Avoidance | ~+20% vs v1 |
| v3 | + Double Buffering | ~+10% vs v2 |

### 关键经验

1. **Memory wall first**: 第一步优化永远是减少 global memory 访问
2. **Register is king**: 尽量在寄存器里计算，shared memory 只做数据分发
3. **Async is free lunch**: SM80+ 的 cp.async 几乎没有额外代价
4. **Know your hardware**: H100、A100、V100 的 shared memory 大小和 bank 结构不同
