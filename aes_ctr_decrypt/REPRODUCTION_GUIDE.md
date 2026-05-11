# 复现指南

本指南记录对话中对 `aes_ctr_decrypt` 算子的所有修改、脚本编写和测试验证过程，方便完整复现。

---

## 1. 环境配置

```
NPU:     Ascend 310P3 (Devices × 4)
CANN:    8.0.0
PyTorch: 2.1.0 + torch_npu
Python:  3.9 (conda env: operator-dev at /root/miniconda3/envs/operator-dev)
```

算子工程根目录：`/root/operator-dev/aes_ctr_decrypt/`

```
├── AesCtrDecrypt/          # 算子实现（Ascend C kernel + host端）
├── AclNNInvocation/        # ACLNN 直接调用方式
├── PytorchInvocation/      # PyTorch 调用方式
├── benchmark_aes_ctr.py    # ACLNN 性能测试（旧）
├── benchmark_large_sizes.py# PyTorch 大尺寸性能测试（本次新增）
├── PERF_ANALYSIS.md        # 性能分析文档（本次新增）
└── REPRODUCTION_GUIDE.md   # 本文件
```

---

## 2. Kernel 修改：解除 16KB 硬编码限制

### 2.1 背景

原始 kernel (`op_kernel/aes_ctr_decrypt.cpp`) 硬编码了 `TOTAL_LENGTH = 8 * 1024` halfs（= 16384 字节），导致单次调用最多只能处理 **16KB** 数据。如需处理更大数据（1MB~2GB），必须修改 kernel 使其支持动态数据量。

### 2.2 修改内容

**文件**: `AesCtrDecrypt/op_kernel/aes_ctr_decrypt.cpp`

**变更 1** — 常量定义替换：

```cpp
// 删除（旧）
constexpr int32_t TOTAL_LENGTH = 8 * 1024;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

// 替换为（新）
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 64;  // 固定 tile 大小，用于 pipeline buffer 分配
```

**变更 2** — `Init()` 方法重写：

`Init()` 签名不变，但内部改为根据 `totalLength` 动态计算每核分块大小：

```cpp
// 关键逻辑
int32_t blockDim = AscendC::GetBlockNum();
int32_t totalHalfs = static_cast<int32_t>(totalLength / 2);
int32_t blockIdx = AscendC::GetBlockIdx();

int32_t baseBlockLen = totalHalfs / blockDim;
int32_t rem = totalHalfs % blockDim;
int32_t startOffset = blockIdx * baseBlockLen + (blockIdx < rem ? blockIdx : rem);
thisBlockLen_ = baseBlockLen + (blockIdx < rem ? 1 : 0);

if (thisBlockLen_ <= 0) { thisBlockLen_ = 0; loopCount_ = 0; return; }
loopCount_ = (thisBlockLen_ + TILE_LENGTH - 1) / TILE_LENGTH;

plaintextGm.SetGlobalBuffer((__gm__ half *)plaintext + startOffset, thisBlockLen_);
ciphertextGm.SetGlobalBuffer((__gm__ half *)ciphertext + startOffset, thisBlockLen_);
```

**变更 3** — `Process()` 改为动态循环：

```cpp
__aicore__ inline void Process() {
    if (thisBlockLen_ <= 0) return;
    for (int i = 0; i < loopCount_; i++) {
        CopyIn(i); Compute(i); CopyOut(i);
    }
}
```

**变更 4** — `CopyIn()`, `Compute()`, `CopyOut()` 支持最后不完整 tile：

新增 `GetTileLen()` 函数：

```cpp
__aicore__ inline int32_t GetTileLen(int32_t idx) {
    int32_t processed = idx * TILE_LENGTH;
    int32_t remaining = thisBlockLen_ - processed;
    return (remaining > TILE_LENGTH) ? TILE_LENGTH : (remaining > 0 ? remaining : TILE_LENGTH);
}
```

各方法中调用 `GetTileLen(idx)` 代替直接使用 `TILE_LENGTH`。

**变更 5** — 新增成员变量：

```cpp
int32_t thisBlockLen_;  // 当前核处理的 half 数量
int32_t loopCount_;     // pipeline 迭代次数
```

### 2.3 完整 diff

修改前后的完整文件可通过以下命令对比：

```bash
# 修改后的文件
cat /root/operator-dev/aes_ctr_decrypt/AesCtrDecrypt/op_kernel/aes_ctr_decrypt.cpp

# 原始文件可参考 git 历史或备份
```

---

## 3. 算子重建与安装

### 3.1 重建

```bash
cd /root/operator-dev/aes_ctr_decrypt/AesCtrDecrypt
rm -rf build_out
bash build.sh
```

预期输出结尾：
```
Self-extractable archive "custom_opp_openEuler_aarch64.run" successfully created.
```

### 3.2 安装

```bash
cd /root/operator-dev/aes_ctr_decrypt/AesCtrDecrypt/build_out
bash custom_opp_openEuler_aarch64.run
```

预期输出结尾：
```
SUCCESS
```

### 3.3 环境变量

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}
```

---

## 4. Kernel 正确性验证

使用 PyTorch 路径验证修正确认 kernel 支持大于 16KB 的数据：

```bash
/root/miniconda3/envs/operator-dev/bin/python3 -c "
import torch, torch_npu
from Crypto.Cipher import AES
import hashlib

key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
iv = torch.zeros((1, 16), dtype=torch.uint8)

for size in [16384, 32768, 65536, 1048576]:
    plaintext = torch.randint(0, 256, size=(size,), dtype=torch.uint8)
    cipher = AES.new(key.numpy().tobytes(), AES.MODE_CTR,
                     initial_value=iv.numpy().tobytes(), nonce=b'')
    golden = torch.frombuffer(bytearray(cipher.encrypt(plaintext.numpy().tobytes())), dtype=torch.uint8)
    output = torch_npu.npu_aes_ctr_decrypt(plaintext.npu(), key.npu()).cpu()
    match = hashlib.md5(output.numpy().tobytes()).hexdigest() == hashlib.md5(golden.numpy().tobytes()).hexdigest()
    print(f'size={size}B: {\"PASS\" if match else \"FAIL\"}')
"
```

预期输出：
```
size=16384B: PASS
size=32768B: PASS
size=65536B: PASS
size=1048576B: PASS
```

---

## 5. 大尺寸性能测试脚本

**文件**: `/root/operator-dev/aes_ctr_decrypt/benchmark_large_sizes.py`

### 5.1 脚本说明

- 使用 PyTorch 调用 `torch_npu.npu_aes_ctr_decrypt()`
- 设置 `jit_compile=False` 避免首次编译开销
- 测试 12 个数据量：1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1G, 2G
- 每次 warmup 3 轮，测量 5 轮取统计值
- 通过 PyCryptodome AES-CTR 验证正确性

### 5.2 运行

```bash
/root/miniconda3/envs/operator-dev/bin/python3 \
  /root/operator-dev/aes_ctr_decrypt/benchmark_large_sizes.py
```

### 5.3 预期结果

| 数据量 | 平均耗时(ms) | 吞吐(MB/s) |
|--------|-------------|-----------|
| 1M     | 13.00      | 76.90     |
| 2M     | 25.59      | 78.15     |
| ...    | ...        | ...       |
| 1G     | 13273.67   | 77.15     |
| 2G     | 26545.99   | 77.15     |

吞吐稳定在 **~77 MB/s**，耗时与数据量呈完美线性关系。

---

## 6. ACLNN 路径的动态测形状适配（附带修改）

修改了 ACLNN C++ 代码以支持从输入文件自动推断 tensor 形状，但该路径最终未用于正式测试（PyTorch 路径更简洁可靠）。

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `AclNNInvocation/src/main.cpp` | `CreateOpDesc()` 改为通过 `GetFileSize()` 获取输入文件大小，动态设置 tensor 形状 |
| `AclNNInvocation/src/common.cpp` | 新增 `GetFileSize()` 函数 |
| `AclNNInvocation/inc/common.h`  | 声明 `GetFileSize()` |

### 重建

```bash
export LD_LIBRARY_PATH=...   # 见 3.3
cd /root/operator-dev/aes_ctr_decrypt/AclNNInvocation
rm -rf build && mkdir build && cd build
cmake ../src && make -j16
```

---

## 7. 之前 Trace 数据的分析

对话第一部分基于 `PytorchInvocation/trace/` 下 16 个 Chrome Trace JSON 文件进行了分析，生成了 `PytorchInvocation/trace/PERF_ANALYSIS.md`。

### 分析内容

- 16 个 trace 文件涵盖多种测试脚本（`benchmark_aes_ctr_encrypt.py`、`test_ops_custom_benchmark.py` 等）
- 首次 JIT 编译开销：78~618ms，稳态 kernel 仅 ~80-190us
- 小数据量（1B~16KB）kernel 耗时主要受 kernel launch 延迟限制

---

## 8. 最终文档

### 8.1 大尺寸性能分析文档

**文件**: `/root/operator-dev/aes_ctr_decrypt/PERF_ANALYSIS.md`

核心发现：
- 吞吐 **~77 MB/s**，大尺寸下稳定线性扩展
- 1GB 解密需约 **13.3 秒**，2GB 约 **26.5 秒**

### 8.2 Trace 分析文档（之前对话）

**文件**: `/root/operator-dev/aes_ctr_decrypt/PytorchInvocation/trace/PERF_ANALYSIS.md`

核心发现：
- 小数据量（1B~16KB）kernel 耗时 ~80-190us
- 首次编译开销 600~5000x
- 数据搬运（H2D/D2H）占 wrapper 总耗时 85%+

---

## 9. 完整复现步骤

```bash
# 1. 激活环境
/root/miniconda3/envs/operator-dev/bin/python3 -c "import torch_npu; print('OK')"

# 2. 修改 kernel
# （修改内容已在上述章节说明，可直接使用仓库中的修改后文件）

# 3. 重建算子
cd /root/operator-dev/aes_ctr_decrypt/AesCtrDecrypt
rm -rf build_out && bash build.sh

# 4. 安装算子
cd build_out && bash custom_opp_openEuler_aarch64.run

# 5. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}

# 6. 验证正确性
# （执行第 4 节的验证命令）

# 7. 运行基准测试
/root/miniconda3/envs/operator-dev/bin/python3 \
  /root/operator-dev/aes_ctr_decrypt/benchmark_large_sizes.py

# 8. 查看报告
cat /root/operator-dev/aes_ctr_decrypt/PERF_ANALYSIS.md
```
