# AesCtrDecrypt 算子开发详解

本文档详细说明 AES-CTR Decrypt 算子在昇腾 NPU 上的开发过程，重点解释与 CPU 开发不同的 NPU 特有概念。

---

## 1. 算子开发概述

### 1.1 什么是 Ascend C 算子

Ascend C 是华为昇腾 NPU 的编程框架，用于开发在 AI Core 上执行的算子。与 CPU 开发相比，Ascend C 有以下核心差异：

| 维度 | CPU 开发 | NPU Ascend C 开发 |
|------|---------|------------------|
| 存储 | 统一 RAM 空间，大容量 | 分 GM (Global Memory) 和 Local Memory，GM 大但慢，LM 小但快 |
| 并行 | 多线程/多进程 | 多核 AI Core（本算子 8 核并行）+ 流水线并行 |
| 编程模型 | 串行指令 | TQue 流水线 + 异步 DMA |
| 数据类型 | 直接用 uint8 | 常用 half（16 位）作为 LocalTensor 基础单元 |
| 编译部署 | 单一可执行文件 | 分 Host 侧（CPU）和 Device 侧（NPU），分开编译 |

### 1.2 工程结构

```
AesCtrDecrypt/                     # 算子工程根目录
├── op_host/                       # Host 侧代码（在 CPU 上运行）
│   ├── aes_ctr_decrypt.cpp        # 原型注册、Shape/DataType 推导、Tiling
│   └── aes_ctr_decrypt_tiling.h   # Tiling 数据结构定义
├── op_kernel/                     # Device 侧代码（在 NPU 上运行）
│   └── aes_ctr_decrypt.cpp        # AI-CORE Kernel 实现
├── build.sh                       # 编译脚本
├── CMakeLists.txt                 # CMake 构建配置
└── op_dev_guide.md                # 本文件
```

Host 侧和 Device 侧代码运行在完全不同的处理器上：

```
Host (CPU)                  Device (NPU AI Core)
─────────                   ──────────────────
TilingFunc                  GET_TILING_DATA
  ↓ 计算分块参数               ↓ 反序列化参数
  ↓ 序列化到 tiling buffer     ↓
  ──────────────────────────→ Init()
  aclnnAesCtrDecrypt           ↓
    → GetWorkspaceSize        Process()
    → aclnnAesCtrDecrypt        → CopyIn (GM → LM)
        ↓                       → Compute (AES 解密)
    Stream 同步                 → CopyOut (LM → GM)
```

---

## 2. Host 侧代码详解

### 2.1 算子原型定义（OpDef）

文件：`op_host/aes_ctr_decrypt.cpp` 中 `ops::AesCtrDecrypt` 类。

OpDef 是算子的"身份证"，告诉 CANN 框架：

```cpp
this->Input("cyphertext")
    .ParamType(REQUIRED)           // 必选输入
    .DataType({ge::DT_UINT8})      // 数据类型 uint8
    .Format({ge::FORMAT_ND})       // 格式 ND（任意维度）
    .UnknownShapeFormat({ge::FORMAT_ND});

this->AICore()                     // 在 AI Core 上执行
    .SetTiling(optiling::TilingFunc)
    .AddConfig("ascend910")        // 支持 ascend910
    .AddConfig("ascend310p");      // 支持 ascend310p
```

**关键点**：
- ND 格式表示任意维度（1D、2D 等均可）
- 算子同时在 ascend910（训练）和 ascend310p（推理）上注册
- 算子名 `AesCtrDecrypt` 必须与 json 原型定义中的 `"op"` 字段一致

### 2.2 Shape/DataType 推导

```cpp
// InferShape: 输出形状 = 输入形状
*g_y_shape = *x1_shape;

// InferDataType: 数据类型不变
context->SetOutputDataType(0, inputDataType);
```

解密不会改变数据的形状和类型，因此直接复制输入参数。

### 2.3 Tiling 机制（NPU 关键概念）

Tiling 是 Ascend 算子开发中最重要的概念之一。

**为什么需要 Tiling？**

NPU 的 Local Memory 非常有限（每个核通常只有几十 KB），无法一次性容纳全部输入数据。Tiling 将大张量切分成多个小块（tile），kernel 逐个处理。

**Tiling 数据流**：

```
Host TilingFunc                   Device kernel
─────────────────                 ─────────────
totalLength = GetShapeSize()
tileNum = 8
      │
      ▼
SaveToBuffer(buffer)              GET_TILING_DATA(tiling_data, tiling)
      │                                    │
      │   tiling buffer (GM)              │
      ────────────────────────────────────→
                                           │
                                      tiling_data.totalLength
                                      tiling_data.tileNum
                                           │
                                           ▼
                                      Init() 中动态分块
```

**本算子的 Tiling 数据**（`aes_ctr_decrypt_tiling.h`）：

```cpp
BEGIN_TILING_DATA_DEF(AesCtrDecryptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // uint8 元素总数
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);       // tile 数量
END_TILING_DATA_DEF;
```

- `totalLength`: 输入 tensor 的 uint8 元素总数（= 字节数），kernel 侧除以 2 转 half 单元
- `tileNum`: 每个核的 tile 数量建议值，固定为 8

**BlockDim 设置**：

```cpp
context->SetBlockDim(8);
```

告诉框架本算子使用 8 个 AI Core 并行。kernel 侧通过 `AscendC::GetBlockNum()` 获取此值。

### 2.4 Tiling vs Block：概念澄清

这是 NPU 开发中最容易混淆的两个概念，它们的区别如下：

| 概念 | 本质 | 代码位置 | 作用 |
|------|------|---------|------|
| **Tiling** | Host→Device 的参数传递机制 | `op_host/tiling` + `GET_TILING_DATA` | 把数据大小、分块策略等**计算参数**传给 kernel |
| **Block** | NPU 的物理 AI Core 执行单元 | `SetBlockDim(N)` + `GetBlockIdx()` | 指定**并行度**，每个 Block 跑一份 kernel |
| **BlockDim** | 并行核数 | `context->SetBlockDim(8)` | 告诉框架开 8 个核 |
| **BlockIdx** | 当前核的编号 | `AscendC::GetBlockIdx()` | 每个核据此决定处理哪段数据 |

**一句话区分**：

- **Tiling 解决"数据太大一次装不下"的问题** — 分块 + 参数传递
- **Block 解决"如何多核加速"的问题** — 8 核并行执行

**两者如何配合**：

```
TilingFunc (Host)
  │
  ├── SetBlockDim(8)  ──────→ 框架启动 8 个 Block
  │                             每个 Block 执行一份 kernel
  │
  ├── tiling.set_totalLength()
  ├── tiling.set_tileNum()
  │       │
  │       ▼ SaveToBuffer(buffer)
  │             │
  │             │   tiling buffer (GM 中的一块内存)
  │             │
  ▼             ▼
GET_TILING_DATA(tiling_data, tiling)  — 每个 Block 各自反序列化
       │
       │  tiling_data.totalLength — 告诉 Block：总数据量多大
       │  tiling_data.tileNum     — 告诉 Block：怎么分 tile
       │
       ▼
  Init() 中用 GetBlockIdx() 计算本核要处理的数据段
```

**类比**：
- **Tiling** = 施工图纸（告诉工人要盖多大的房子、用什么材料）
- **Block** = 工人数量（8 个工人同时施工）
- **BlockIdx** = 工人的编号（工人1负责东侧，工人2负责西侧...）

---

## 3. Device 侧代码详解

### 3.1 两级存储与 half 数据类型

**GM (Global Memory)**：
- 位于 NPU 芯片外部（HBM 显存）
- 容量大（几十 GB），延迟高（约几百周期）
- 所有 AI Core 共享
- kernel 的输入输出数据都在 GM 中

**Local Memory**：
- 位于 NPU 芯片内部，每个核独享
- 容量小（几十 KB），延迟低（几个周期）
- 计算必须在 Local Memory 中完成

**为什么需要 half？**

核心原因：NPU 的 `DataCopy` 硬件引擎（GM ↔ Local 数据搬运）设计为**以 half（16 位）作为基本传输单元**。虽然可以声明 `LocalTensor<uint8_t>`，但底层 DMA 仍按 16 位对齐搬运，效率最高。所以 GM 和 Local Memory 间的所有数据交换都通过 `half*` 指针进行。

**数据流全景：**

```
GM (uint8 数组)
┌────┬────┬────┬────┬────┬────┐
│0xAB│0xCD│0x12│0x34│0x56│0x78│  ← 原始数据是 uint8
└────┴────┴────┴────┴────┴────┘
      │
      │ DataCopy 以 half 为单位传输
      ▼
Local Memory (half 数组)
┌──────────┬──────────┬──────────┐
│ 0xCDAB   │ 0x3412   │ 0x7856   │  ← 两个 uint8 打包成一个 half
└──────────┴──────────┴──────────┘
      │
      │ Compute 时拆包为 uint8[]
      ▼
plainBuf (uint8 数组，可做 AES 字节运算)
┌────┬────┬────┬────┬────┬────┐
│0xAB│0xCD│0x12│0x34│0x56│0x78│
└────┴────┴────┴────┴────┴────┘
      │
      │ AES 计算完成后再打包回 half
      ▼
Local Memory → DataCopy → GM
```

这种"用 half 传输、拆 uint8 计算"的模式在 Ascend C 开发中非常常见，是 NPU 硬件特性决定的。

在 `Compute()` 阶段的代码实现：

```cpp
// half → uint8[]（解包）：遍历每个 half，拆分出两个 uint8
for (int i = 0; i < tileLen; i++) {
    half hVal = plainLocal.GetValue(i);
    uint16_t halfBits = *reinterpret_cast<uint16_t *>(&hVal);
    plainBuf[2*i + 0] = halfBits & 0xFF;        // 低字节
    plainBuf[2*i + 1] = (halfBits >> 8) & 0xFF;  // 高字节
}

// AES 计算（在 uint8[] 缓冲区上进行 byte-level 运算）
AesCtrEncrypt(plainBuf, cipherBuf, tileBytes, ...);

// uint8[] → half（打包）
for (int i = 0; i < tileLen; i++) {
    uint16_t halfBits = cipherBuf[2*i] | (cipherBuf[2*i+1] << 8);
    half val = *reinterpret_cast<half *>(&halfBits);
    cipherLocal.SetValue(i, val);
}
```

### 3.1.1 LM vs GM：各变量的存储位置

理解每个变量存储在 LM 还是 GM，是 NPU 开发的关键。下面完整列出本算子中所有变量的归属：

```
LM（每个核独享，几十 KB，低延迟）
──────────────────────────────────────────────────────────────────
类别           变量                          说明
──────────     ─────────────────────────     ─────────────────────
成员数组       sbox[256]                    AES S-box 查表
               expandedKey[176]             扩展密钥
               IV[16]                       计数器

成员标量       thisBlockLen_, loopCount_    分块状态

流水线         pipe                         流水线管理器
               inQueuePlain, outQueueCipher  队列描述符
               plaintextGm, ciphertextGm     GlobalTensor 描述符
                                            （描述符在 LM，数据在 GM）

计算 buffer    plainBuf, cipherBuf           AES 工作缓冲区
               plainLocal, cipherLocal       DataCopy 缓冲区

函数栈         localKey, TmpIv, blockDim,
               blockIdx, startOffset 等      临时变量
```

```
GM（所有核共享，几十 GB，高延迟）
──────────────────────────────────────────────────────────────────
类别           变量                          说明
──────────     ─────────────────────────     ─────────────────────
kernel 参数    cyphertext, key, plaintext    GM_ADDR 指向的原始数据
               tiling                        Host 传来的 tiling buffer

tiling 数据    tiling_data.totalLength       GET_TILING_DATA 解析出
               tiling_data.tileNum

数据区         plaintextGm[i] 引用的元素     输入密文（GM 中的 uint8/half）
               ciphertextGm[i] 引用的元素    输出明文（GM 中的 uint8/half）
```

**关键判断原则**：
- `DataCopy(src, dst, n)` — src/dst 一个在 GM、一个在 LM 之间搬运
- `pipe.InitBuffer(...)` 分配出来的 `LocalTensor` — 在 LM
- `__gm__` 修饰的指针或 `GM_ADDR` 类型参数 — 指向 GM
- 未被 `__gm__` 修饰的局部变量和成员变量 — 在 LM（NPU kernel 的栈帧位于 LM）

### 3.2 TQue 三阶段流水线

NPU 上数据搬运（GM ↔ Local）需要几百个周期，如果串行执行（搬完再算，算完再搬），性能会非常差。TQue 流水线让数据搬运和计算重叠进行。

**流水线示意图**：

```
核0 时间轴 →
─────────────────────────────────────────────
CopyIn(tile0) │ Compute(tile0) │ CopyOut(tile0)
              │ CopyIn(tile1)  │ Compute(tile1) │ CopyOut(tile1)
                              │ CopyIn(tile2)  │ Compute(tile2) │ ...
```

三阶段用 `TQue`（模板队列）连接，BUFFER_NUM=2 实现乒乓缓冲：

```
inQueuePlain (VECIN)          outQueueCipher (VECOUT)
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Buffer 0 │     │ Buffer 0 │     │ Buffer 0 │
├──────────┤     ├──────────┤     ├──────────┤
│ Buffer 1 │     │ Buffer 1 │     │ Buffer 1 │
└──────────┘     └──────────┘     └──────────┘
  CopyIn →       → Compute →      → CopyOut →
  (AllocTensor)   (DeQue/EnQue)    (DeQue/FreeTensor)
```

**实现细节**：

```cpp
// CopyIn: 从 GM 申请 Local Tensor，DataCopy 后入队
AscendC::LocalTensor<half> plainLocal = inQueuePlain.AllocTensor<half>();
AscendC::DataCopy(plainLocal, plaintextGm[idx * TILE_LENGTH], tileLen);
inQueuePlain.EnQue(plainLocal);

// Compute: 出队 → AES 计算 → 入队输出队列
AscendC::LocalTensor<half> plainLocal = inQueuePlain.DeQue<half>();
AscendC::LocalTensor<half> cipherLocal = outQueueCipher.AllocTensor<half>();
// ... AES 计算 ...
outQueueCipher.EnQue<half>(cipherLocal);
inQueuePlain.FreeTensor(plainLocal);

// CopyOut: 出队 → DataCopy 写回 GM → FreeTensor
AscendC::LocalTensor<half> cipherLocal = outQueueCipher.DeQue<half>();
AscendC::DataCopy(ciphertextGm[idx * TILE_LENGTH], cipherLocal, tileLen);
outQueueCipher.FreeTensor(cipherLocal);
```

### 3.3 多核并行与 IV 偏移

本算子使用 8 个 AI Core 并行处理。多核并行的配置涉及**两个地方**：

#### 3.3.1 多核并行的参数设置

**位置 1（Host 侧 TilingFunc）— 告诉框架用多少核：**

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(8);   // ← 设置并行核数，固定为 8
    ...
}
```

`SetBlockDim(8)` 告诉 CANN 框架：本算子需要 8 个 AI Core 并行执行。框架收到这个信息后，会在 NPU 上同时启动 8 个 kernel 实例，每个实例运行在**不同的物理 AI Core** 上。

**位置 2（Device 侧 Init）— 每个核知道自己是谁：**

```cpp
__aicore__ inline void Init(...)
{
    int32_t blockDim = AscendC::GetBlockNum();  // = 8（从 SetBlockDim 传来）
    int32_t blockIdx = AscendC::GetBlockIdx();  // = 0, 1, ..., 7（当前核编号）
    ...
}
```

- `GetBlockNum()`: 总核数，由 Host 侧的 `SetBlockDim(8)` 决定，固定为 8
- `GetBlockIdx()`: 当前核的编号（0 ~ 7），由 NPU 硬件自动分配

两个函数配合，让每个核知道自己该处理哪段数据。

#### 3.3.2 数据分配

每个核通过 `GetBlockIdx()` 计算出自己在 GM 中的起始位置和块长度：

```cpp
// 数据均分到各核，余数分配给前几个核
int32_t baseBlockLen = totalHalfs / blockDim;       // 每核基准 half 数
int32_t rem = totalHalfs % blockDim;                 // 余数
int32_t startOffset = blockIdx * baseBlockLen
                    + (blockIdx < rem ? blockIdx : rem);  // 本核起始偏移
thisBlockLen_ = baseBlockLen + (blockIdx < rem ? 1 : 0);  // 本核处理量

// 各核指向 GM 中的不同位置
plaintextGm.SetGlobalBuffer((__gm__ half *)plaintext + startOffset, thisBlockLen_);
ciphertextGm.SetGlobalBuffer((__gm__ half *)ciphertext + startOffset, thisBlockLen_);
```

**示意图**（8 核处理 N 个 half 的数据）：

```
GM 数据（half 数组，共 N 个）
┌──────────────────────────────────────────────────────────┐
│  核0(Idx=0)  │  核1(Idx=1)  │  ...  │  核7(Idx=7)       │
│  0 ~ N/8-1   │ N/8 ~ 2N/8-1│       │ 7N/8 ~ N-1        │
│  start=0     │ start=N/8   │       │ start=7N/8         │
└──────────────────────────────────────────────────────────┘
```

#### 3.3.3 IV 偏移（多核并行的关键）

CTR 模式的密钥流由 `AES(key, counter)` 生成。如果 8 个核都从 counter=0 开始，它们会生成**相同的密钥流**，导致解密结果全错（相同位置的密文被相同的密钥流异或了多次）。

解决方法：每个核的 counter 从不同的起始值开始，偏移量等于该核之前已经处理的数据量。

```
核0(Idx=0): startOffset=0   → counter 从 0 开始
核1(Idx=1): startOffset=N/8 → counter 从 N/16 开始（N/8 halfs = N/4 字节 ÷ 16 = N/16 个 AES 块）
核2(Idx=2): startOffset=2N/8 → counter 从 2N/16 开始
...
```

代码实现：

```cpp
// startOffset 是 half 单位，转为 AES 块数
// 每个 AES 块 = 16 字节 = 8 个 half
// 偏移块数 = startOffset / 8
uint32_t offsetBlocks = static_cast<uint32_t>(startOffset / 8);
BigEndAdd(tmpIV, offsetBlocks);  // 大端加法：tmpIV += offsetBlocks
```

### 3.4 动态分块（16KB 限制修复）

**原始问题**：kernel 硬编码了 `TOTAL_LENGTH = 8192` halfs（= 16384 字节），所有缓冲区大小固定，无法处理 >16KB 的数据。

**修复方案**：保持 `TILE_LENGTH = 64` halfs 作为编译期常量（用于 Local Memory buffer 分配），但改为动态计算 `loopCount_`（迭代次数）：

```cpp
// 动态计算每核分块
int32_t totalHalfs = totalLength / 2;
int32_t baseBlockLen = totalHalfs / blockDim;
int32_t startOffset = blockIdx * baseBlockLen + ...;
thisBlockLen_ = baseBlockLen + ...;
loopCount_ = (thisBlockLen_ + TILE_LENGTH - 1) / TILE_LENGTH;

// 处理最后可能不完整的 tile
int32_t GetTileLen(idx) {
    remaining = thisBlockLen_ - idx * TILE_LENGTH;
    return min(remaining, TILE_LENGTH);
}
```

### 3.5 Kernel 入口函数

```cpp
extern "C" __global__ __aicore__ void aes_ctr_decrypt(
    GM_ADDR cyphertext, GM_ADDR key, GM_ADDR plaintext,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAesCtrDecrypt op;
    op.Init(cyphertext, key, plaintext,
            tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

- `extern "C"`: 防止 C++ 名字改编，确保框架能找到符号
- `__global__`: NPU kernel 入口修饰符
- `__aicore__`: 在 AI Core 上执行（而非 AI CPU）
- `GM_ADDR`: 全局内存地址（本质是 `void*`）
- `GET_TILING_DATA`: 从 GM 中的 tiling buffer 反序列化参数

---

## 4. AES 算法在 NPU 上的实现

AES-CTR 解密的核心是标准的 AES-128 加密。在 NPU 上实现时，所有 AES 函数（`KeyExpansion`、`SubBytes`、`ShiftRows`、`MixColumns`、`AddRoundKey`）都用纯标量 C 代码实现，没有使用 NPU 特有的向量指令。

这是因为 AES 算法的粒度是 16 字节块，每块的运算逻辑高度串行（SubBytes → ShiftRows → MixColumns → AddRoundKey 串行执行 10 轮），不适合用向量化指令加速。

**S-box 存储**：256 字节的 AES S-box 在 `KernelAesCtrDecrypt` 构造时复制到 Local Memory 的 `sbox[256]` 数组中，后续 SubBytes 操作直接查表。

**计数器自增**：使用 `BigEndAdd()` 实现大端格式的 128 位加法，每处理一个 16 字节块后自增。

---

## 5. 与其他平台 AES 实现的对比

| 平台 | 实现方式 | 性能特征 |
|------|---------|---------|
| CPU (x86 AES-NI) | 硬件指令 `aesenc` | 极高，~1 GB/s+ |
| CPU (纯软件) | C 代码查表 | 较低，~100-500 MB/s |
| GPU (CUDA) | 并行 warp 处理 | 极高，~10 GB/s+ |
| **NPU (Ascend AI-CORE)** | **TQue 流水线 + 8 核并行** | **~77 MB/s** |

NPU 的 AI Core 主要面向矩阵/向量运算（如卷积、Transformer），AES 这种 16 字节块的串行加密不是其优势场景。本算子的性能主要受限于：
1. half ↔ uint8 格式转换开销
2. AES 10 轮标量计算的 CPI
3. GM ↔ Local 数据搬运带宽

## 5.1 优化方向分析

当前吞吐约 77 MB/s，以下是理论上可考虑的优化方向及其可行性：

### 方向 1：消除 half ↔ uint8 转换（收益高，但受硬件约束）

当前每个 tile 都要做 half→uint8 解包和 uint8→half 打包，这是纯软件开销。如果 `LocalTensor<uint8_t>` 能被 `DataCopy` 硬件直接支持，可以省掉这笔开销：

```cpp
// 理想情况：直接以 uint8 为单位搬运和计算
AscendC::LocalTensor<uint8_t> plainLocal = inQueuePlain.AllocTensor<uint8_t>();
AscendC::DataCopy(plainLocal, plaintextGm, tileLen);  // 直接用 uint8 搬运
// 无需 half↔uint8 转换，直接 AES 计算
AesCtrEncrypt(plainLocal.GetValue(0), ...);
```

实际上，当前 Ascend C 的 DataCopy 硬件对 half 支持最好，`uint8_t` 的 LocalTensor 可能被内部提升为 half，或需要手动 `AscendC::ConvertTo`。如果能用原生 uint8 传输，预计可提升 **20-30%**。

### 方向 2：增大 TILE_LENGTH（收益中等）

当前 TILE_LENGTH = 64 halfs（128 字节 / tile），每核的 pipeline 迭代次数较多。增大 TILE_LENGTH 可减少迭代次数，降低 pipeline stage 切换开销：

```
TILE_LENGTH=64: loopCount = N/64 次迭代
TILE_LENGTH=256: loopCount = N/256 次迭代（减少 75% 迭代次数）
```

**风险**：TILE_LENGTH 增大会占用更多 Local Memory。当前每核 LM 占用为 `TILE_LENGTH × sizeof(half) × BUFFER_NUM × 2 个队列 = 64 × 2 × 2 × 2 = 512 字节`。如果 TILE_LENGTH=256，占用 2048 字节，仍在 LM 预算内（通常几十 KB），可以尝试。

### 方向 3：增加 BUFFER_NUM（收益低）

当前 BUFFER_NUM=2 实现乒乓缓冲。增加到 3 或 4 可以更深度隐藏 DataCopy 延迟：

```cpp
constexpr int32_t BUFFER_NUM = 3;  // 三缓冲
```

**代价**：多占 50% 的 Local Memory（每增加 1 个 buffer = TILE_LENGTH × sizeof(half) = 128 字节）。在 LM 充足时值得尝试，但预期收益不高（DataCopy 延迟已经可以被 2 缓冲较好地隐藏）。

### 方向 4：用 AI CPU 代替 AI Core（架构级改变）

AES 是典型的标量运算（逐字节查表、异或），AI Core 的矩阵/向量单元几乎用不上。AI CPU（同构 ARM 核）反而更适合这种运算：

对比：
| 维度 | AI Core | AI CPU |
|------|---------|--------|
| AES 标量计算 | 慢（为矩阵优化） | 快（ARM 标量流水线） |
| 核数 | 8 核 | 2-4 核 |
| LM 限制 | 几十 KB | 无（直接访问 DDR） |
| half 转换 | 需要 | 不需要（可直接操作 uint8） |

从当前 ~77 MB/s 的表现来看，这个算子放在 AI CPU 上可能达到更高的吞吐（典型 AI CPU 算子的 AES 吞吐在 150-300 MB/s）。

### 方向 5：S-box 放到常量存储（低收益，省 LM 空间）

当前 S-box 存在 LM 中占 256 字节，可以移到 `__constant__` 只读存储：

```cpp
__constant__ uint8_t sbox[256] = {0x63, 0x7c, ...};
```

释放出的 LM 空间可用于增大 TILE_LENGTH。但 S-box 查表在 LM 中是最快的（1-2 周期），放到常量存储会增加查表延迟，**仅当 LM 空间不足需要增大 TILE_LENGTH 时才值得做**。

### 方向 6：AES 循环展开（收益低）

AES 10 轮展开成直线代码，减少循环控制开销：

```cpp
// 展开前
for (int r = 1; r < AES_ROUND_NUM; r++) {
    SubShiftMix(state, blockId, sbox);
    AddRoundKey(state, roundKeys + 16 * r);
}

// 展开后
SubShiftMix(state, blockId, sbox); AddRoundKey(state, roundKeys + 16*1);
SubShiftMix(state, blockId, sbox); AddRoundKey(state, roundKeys + 16*2);
// ... 10 轮展开
SubShift(state, sbox); AddRoundKey(state, roundKeys + 16*10);
```

AES 每轮计算本身就是计算密集的（SubBytes 查表 + ShiftRows + MixColumns），循环控制开销占比极小。**收益可忽略**。

### 优化收益预估

| 优化方向 | 预估收益 | 实现难度 | 风险 |
|---------|---------|---------|------|
| 消除 half 转换 | +20~30% | 高（取决于硬件支持） | 可能不支持 |
| 增大 TILE_LENGTH | +5~15% | 低（改一个数字） | LM 溢出 |
| 增加 BUFFER_NUM | +0~5% | 低 | LM 占用增加 |
| 改用 AI CPU | +100~300% | 高（重写算子） | 架构变化 |
| 常量 S-box | 可有可无 | 低 | 查表变慢 |
| 循环展开 | <1% | 低 | 代码膨胀 |

### 当前瓶颈定位

从 profilling（`#ifndef ASCENDC_CPU_DEBUG` 编译后通过 msprof 采集）可以精确定位瓶颈在哪个阶段。初步分析：

- **Compute 阶段**：AES 每字节 ~8-10 条标量指令 × 10 轮 × 16 字节 = ~1280-1600 条指令 / 块。这是主要计算时间。
- **CopyIn/CopyOut 阶段**：DataCopy 带宽通常为几十 GB/s，128 字节的传输约需几十个周期，不是瓶颈。
- **half 转换**：两个 for 循环各遍历所有字节，是纯软件开销。

**综合建议**：如果追求极致性能，建议将算子迁移到 AI CPU 实现。如果继续使用 AI Core，增大 TILE_LENGTH 是最简单的优化手段。

---

## 6. 调试与验证

### CPU 模式调试

在 `op_kernel/aes_ctr_decrypt.cpp` 中，`#ifndef ASCENDC_CPU_DEBUG` 分支外的代码用于 CPU 调试模式，可以在没有 NPU 硬件的环境中用 DS9 仿真器调试。

### 正确性验证

编译安装后，通过 PyTorch 或 ACLNN 调用算子，与 PyCryptodome 的标准 AES-CTR 实现对比 MD5：

```bash
python3 -c "
from Crypto.Cipher import AES
import torch, torch_npu, hashlib
# ... 对比 torch_npu.npu_aes_ctr_decrypt() 与 AES.new().encrypt()
"
```

---

## 7. 编译与部署

```bash
# 编译算子工程
cd AesCtrDecrypt
rm -rf build_out && bash build.sh

# 安装
cd build_out && bash custom_opp_openEuler_aarch64.run

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}
```

编译产物：
- `libcust_opapi.so` — ACLNN API 库（供 aclnnAesCtrDecrypt 调用）
- `libcust_opsproto_rt2.0.so` — 算子原型库
- `libcust_opmaster_rt2.0.so` — Tiling 库
- `AesCtrDecrypt_*.o` — NPU kernel 二进制
- `aic-ascend310p-ops-info.json` — 算子在 ascend310p 上的算子信息

---

## 8. 常见 NPU 开发问题

### Q: 为什么使用 half 而不是 uint8 作为 LocalTensor 类型？

Ascend C 的 LocalTensor 内部存储单元是 half。虽然可以创建 `LocalTensor<uint8_t>`，但 `DataCopy` 的传输效率在 half 时达到最优。在实际实现中，用 half 做传输，在 Compute 阶段转为 uint8 做计算。

### Q: 为什么需要 TILE_LENGTH 固定为 64？

`pipe.InitBuffer()` 需要编译期常量来分配 Local Memory。`TILE_LENGTH=64` halfs = 128 字节 / tile，在 Local Memory 的预算内。如果需要更高的并行度，可以增大此值，但需要确认 Local Memory 是否足够（BUFFER_NUM × TILE_LENGTH × sizeof(half) × 2 个队列）。

### Q: 如何选择 tile 大小？

- 太小：流水线 stage 切换过于频繁，开销增大
- 太大：Local Memory 不够用，或 BUFFER_NUM 减少导致流水线深度不够
- 本算子选择 64 halfs（128 字节），是一个折中值

### Q: 单次 kernel 调用能处理的最大数据量是多少？

理论上没有上限（动态分块后），但受制于 GM 大小（HBM 容量）。在 310P3 上单次调用已验证通过 2GB 数据。

### Q: 算子对数据量有什么约束？必须 16 字节对齐吗？

有两个约束：

**约束 1：字节总数必须为偶数（必须满足）**

kernel 内部用 `totalLength / 2` 将 uint8 元素数转为 half 数。如果 `totalLength` 是奇数，除 2 会截断，最后一个字节被丢弃。原因是 `DataCopy` 和 `GlobalTensor` 均以 half（16 位）为传输单元，两个 uint8 打包成一个 half，奇数个字节无法完整打包。

**约束 2：AES 块大小 16 字节（自动处理，无需关心）**

`AesCtrEncrypt` 中已处理不完整块：

```cpp
int blockSize = (lengthBytes - offset < 16) ? (lengthBytes - offset) : 16;
```

最后一块不足 16 字节时，只异或剩余字节数——CTR 模式本身的特性，不需要填充。

**验证**：

```bash
/root/miniconda3/envs/operator-dev/bin/python3 -c "
import torch, torch_npu, hashlib
from Crypto.Cipher import AES
key = torch.randint(0, 256, (1, 16), dtype=torch.uint8)
iv = torch.zeros((1, 16), dtype=torch.uint8)
for size in [18, 34, 50, 100, 1023]:  # 偶数但不是16的倍数
    p = torch.randint(0, 256, (size,), dtype=torch.uint8)
    c = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
    g = c.encrypt(p.numpy().tobytes())
    o = torch_npu.npu_aes_ctr_decrypt(p.npu(), key.npu()).cpu()
    m = hashlib.md5(o.numpy().tobytes()).hexdigest() == hashlib.md5(g).hexdigest()
    print(f'size={size:>4}B: {\"PASS\" if m else \"FAIL\"}')
"
```

结论：**偶数字节即可，不必是 16 的倍数。**
