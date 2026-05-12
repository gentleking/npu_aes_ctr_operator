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
└── OP_DEV_GUIDE.md                # 本文件
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

**half 数据类型的使用**：

Ascend C 的 `LocalTensor<T>` 使用 `half`（16 位浮点）作为基本存储单元。但 AES 处理的是 8 位字节数据。解决方法是将两个 uint8 打包成一个 half：

```
uint8[0]  uint8[1]           half
0xAB      0xCD      →    0xCDAB (16-bit)
↑低字节    ↑高字节
```

在 `Compute()` 阶段：

```cpp
// half → uint8[]（解包）
for (int i = 0; i < tileLen; i++) {
    half hVal = plainLocal.GetValue(i);
    uint16_t halfBits = *reinterpret_cast<uint16_t *>(&hVal);
    plainBuf[2*i + 0] = halfBits & 0xFF;        // 低字节
    plainBuf[2*i + 1] = (halfBits >> 8) & 0xFF;  // 高字节
}

// AES 计算（在 uint8[] 缓冲区上进行）
AesCtrEncrypt(plainBuf, cipherBuf, tileBytes, ...);

// uint8[] → half（打包）
for (int i = 0; i < tileLen; i++) {
    uint16_t halfBits = cipherBuf[2*i] | (cipherBuf[2*i+1] << 8);
    half val = *reinterpret_cast<half *>(&halfBits);
    cipherLocal.SetValue(i, val);
}
```

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

本算子使用 8 个 AI Core 并行处理。每个核处理一部分数据，关键是**每个核必须使用不同的密钥流**。

**数据分配**：

```
GM 数据（half 数组）
┌──────────────────────────────────────────────┐
│  核0数据  │  核1数据  │  ...  │  核7数据  │
│  0~N/8   │ N/8~2N/8  │       │ 7N/8~N   │
└──────────────────────────────────────────────┘
```

每个核的块大小 = `totalHalfs / 8`，余数分配给前几个核。

**IV 偏移计算**：

CTR 模式的密钥流由 AES(key, counter) 生成。如果 8 个核都从 counter=0 开始，它们会生成相同的密钥流，导致解密错误。

解决方法：按核的起始数据位置偏移 counter。

```
核0 从数据偏移 0 开始 → counter 从 0 开始
核1 从数据偏移 N/8 开始 → counter 从 N/16 开始（因为每 16 字节 = 1 个 counter）
核2 从数据偏移 2N/8 开始 → counter 从 2N/16 开始
...
```

代码实现：

```cpp
// startOffset 是 half 单位，转为 AES 块数
// 每个 AES 块 = 16 字节 = 8 个 half
// 所以偏移块数 = startOffset / 8
uint32_t offsetBlocks = static_cast<uint32_t>(startOffset / 8);
BigEndAdd(tmpIV, offsetBlocks);
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
