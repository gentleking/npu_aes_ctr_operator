# AES-CTR Decrypt 算子性能分析报告 & 复现指南

---

## 第一部分：性能分析

### 测试环境

- **硬件**: Ascend 310P3 (4 devices, 1 used)
- **显存**: 44 GB / device
- **CANN**: 8.0.0
- **PyTorch**: 2.1.0, torch_npu
- **算子类型**: AI-CORE (Ascend C, TQue Pipeline, 8 core)
- **编译模式**: `jit_compile=False`

### 测试方法

- 使用 `torch_npu.npu_aes_ctr_decrypt()` 进行解密
- 数据量从 1MB 到 2GB，uint8 类型，1D tensor
- 每次测试前 3 轮 warmup，取 5 轮有效测量
- 统计平均耗时、最小耗时、最大耗时和吞吐量
- 加密密钥固定 16 字节 uint8，IV 为全零
- 通过 PyCryptodome AES-CTR 参考实现验证结果正确性

### 测试结果

| 数据量 | 平均耗时(ms) | 最小(ms) | 最大(ms) | 吞吐(MB/s) |
|--------|-------------|---------|---------|-----------|
| 1M     | 13.00      | 12.94   | 13.06   | 76.90     |
| 2M     | 25.59      | 25.58   | 25.61   | 78.15     |
| 4M     | 51.10      | 51.09   | 51.14   | 78.27     |
| 8M     | 102.14     | 102.10  | 102.19  | 78.32     |
| 16M    | 207.92     | 207.88  | 208.04  | 76.95     |
| 32M    | 415.44     | 415.36  | 415.58  | 77.03     |
| 64M    | 830.68     | 830.62  | 830.76  | 77.05     |
| 128M   | 1661.73    | 1661.64 | 1661.85 | 77.03     |
| 256M   | 3321.38    | 3320.90 | 3321.73 | 77.08     |
| 512M   | 6638.62    | 6636.68 | 6640.47 | 77.12     |
| 1G     | 13273.67   | 13265.15 | 13278.93 | 77.15   |
| 2G     | 26545.99   | 26535.23 | 26565.01 | 77.15   |

### 性能特征总结

1. **线性扩展性极好**：耗时与数据量呈完美线性关系，吞吐稳定在 **~77 MB/s**。拟合模型：`time(ms) ≈ 12.97ms per MB`。
2. **延迟微秒级**：小数据量（<1MB）时 kernel launch 启动开销占主导（~80-190us），大数据量时计算主导。
3. **无 kernel launch 瓶颈**：单次 kernel launch 即可处理 2GB。
4. **首次编译开销**：`jit_compile=True` 时约 80-600ms；`jit_compile=False` 无编译开销。

不同数据量耗时参考：

| 数据量 | 耗时 |
|--------|------|
| 1 KB | ~0.01ms |
| 1 MB | ~13ms |
| 10 MB | ~130ms |
| 100 MB | ~1.3s |
| 1 GB | ~13.3s |
| 2 GB | ~26.5s |

### Trace 数据分析

`PytorchInvocation/trace/` 目录下 16 个 Chrome Trace 文件的 profiling 结果：

- **首次 JIT 编译开销**：78~618ms（环境差异较大），稳态后 kernel 仅 ~80-190us，编译开销约 600~5000x
- **数据搬运开销**：完整 wrapper 调用（H2D + kernel + D2H）中，传输开销占 85-90%，kernel 仅占 10-15%
- **小数据量特性**：1B~16KB 范围内 kernel 耗时 ~80-190us，kernel launch 启动开销占主导

---

## 第二部分：复现指南

本部分记录对算子的所有修改、脚本编写和测试验证过程，方便完整复现。

### 环境配置

```
NPU:     Ascend 310P3 (Devices × 4)
CANN:    8.0.0
PyTorch: 2.1.0 + torch_npu
Python:  3.9 (conda env: operator-dev at /root/miniconda3/envs/operator-dev)
```

算子工程目录结构：

```
/root/operator-dev/aes_ctr_decrypt/
├── AesCtrDecrypt/           # 算子实现（Ascend C kernel + host端）
│   ├── build.sh             # 编译入口
│   ├── CMakeLists.txt
│   ├── op_host/
│   │   ├── aes_ctr_decrypt.cpp       # 原型注册、tiling
│   │   └── aes_ctr_decrypt_tiling.h  # tiling 数据定义
│   └── op_kernel/
│       └── aes_ctr_decrypt.cpp       # 核函数实现（已修改）
├── AclNNInvocation/         # ACLNN 直调模式
│   ├── src/main.cpp         # 主程序（已修改为动态形状）
│   ├── src/common.cpp       # 工具函数（新增 GetFileSize）
│   └── inc/common.h         # 头文件
├── PytorchInvocation/       # PyTorch 调用模式
│   ├── op_plugin_patch/     # PTA 适配补丁
│   ├── test_ops_custom.py
│   ├── test_ops_custom_benchmark.py
│   ├── benchmark_aes_ctr_encrypt.py
│   └── run_op_plugin.sh
├── benchmark_all.py          # 统一性能测试脚本（1KB~2GB）
├── perf_analysis.md          # 本文件（性能分析 + 复现指南）
└── readme.md                 # 项目说明
```

### Kernel 修改：解除 16KB 硬编码限制

**背景**：原始 kernel 硬编码了 `TOTAL_LENGTH = 8 * 1024` halfs（= 16384 字节），无法处理大于 16KB 的数据。

**文件**: `AesCtrDecrypt/op_kernel/aes_ctr_decrypt.cpp`

**修改内容**：

**变更 1** — 常量定义替换：

```cpp
// 删除
constexpr int32_t TOTAL_LENGTH = 8 * 1024;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;

// 保留
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 64;  // 固定用于 buffer 分配
```

**变更 2** — `Init()` 动态计算每核分块：

```cpp
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

**变更 3** — `Process()` 动态循环：

```cpp
__aicore__ inline void Process() {
    if (thisBlockLen_ <= 0) return;
    for (int i = 0; i < loopCount_; i++) {
        CopyIn(i); Compute(i); CopyOut(i);
    }
}
```

**变更 4** — 新增 `GetTileLen()` 处理不完整 tile：

```cpp
__aicore__ inline int32_t GetTileLen(int32_t idx) {
    int32_t processed = idx * TILE_LENGTH;
    int32_t remaining = thisBlockLen_ - processed;
    return (remaining > TILE_LENGTH) ? TILE_LENGTH : (remaining > 0 ? remaining : TILE_LENGTH);
}
```

**变更 5** — 新增成员变量：

```cpp
int32_t thisBlockLen_;  // 当前核处理的 half 数量
int32_t loopCount_;     // pipeline 迭代次数
```

### 算子重建与安装

```bash
# 重建
cd /root/operator-dev/aes_ctr_decrypt/AesCtrDecrypt
rm -rf build_out
bash build.sh

# 安装
cd build_out
bash custom_opp_openEuler_aarch64.run

# 环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}
```

### ACLNN 路径修改（附带）

修改 ACLNN 代码支持从输入文件大小自动推断 tensor 形状：

| 文件 | 修改内容 |
|------|---------|
| `AclNNInvocation/src/main.cpp` | `CreateOpDesc()` 通过 `GetFileSize()` 动态设置 tensor 形状 |
| `AclNNInvocation/src/common.cpp` | 新增 `GetFileSize()` 函数 |
| `AclNNInvocation/inc/common.h` | 声明 `GetFileSize()` |

```bash
# 重建 ACLNN 可执行文件
export LD_LIBRARY_PATH=...  # 同上
cd /root/operator-dev/aes_ctr_decrypt/AclNNInvocation
rm -rf build && mkdir build && cd build
cmake ../src && make -j16
```

### Kernel 正确性验证

```bash
/root/miniconda3/envs/operator-dev/bin/python3 -c "
import torch, torch_npu
from Crypto.Cipher import AES; import hashlib
key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
iv = torch.zeros((1, 16), dtype=torch.uint8)
for size in [16384, 32768, 65536, 1048576]:
    p = torch.randint(0, 256, size=(size,), dtype=torch.uint8)
    c = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
    g = torch.frombuffer(bytearray(c.encrypt(p.numpy().tobytes())), dtype=torch.uint8)
    o = torch_npu.npu_aes_ctr_decrypt(p.npu(), key.npu()).cpu()
    m = hashlib.md5(o.numpy().tobytes()).hexdigest() == hashlib.md5(g.numpy().tobytes()).hexdigest()
    print(f'size={size}B: {\"PASS\" if m else \"FAIL\"}')
"
```

预期输出：全部 PASS。

### 运行基准测试

统一测试脚本 `benchmark_all.py`（覆盖 1KB ~ 2GB，共 17 个量级）：

```bash
/root/miniconda3/envs/operator-dev/bin/python3 \
  /root/operator-dev/aes_ctr_decrypt/benchmark_all.py
```

预期结果：吞吐稳定 ~77 MB/s，所有量级 PASS。

### 完整复现步骤

```bash
# 1. 激活环境
/root/miniconda3/envs/operator-dev/bin/python3 -c "import torch_npu; print('OK')"

# 2. 修改 kernel（内容见上，或直接使用仓库代码）

# 3. 重建算子
cd /root/operator-dev/aes_ctr_decrypt/AesCtrDecrypt
rm -rf build_out && bash build.sh

# 4. 安装算子
cd build_out && bash custom_opp_openEuler_aarch64.run

# 5. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}

# 6. 验证正确性（执行上面的验证命令）

# 7. 运行基准测试
/root/miniconda3/envs/operator-dev/bin/python3 /root/operator-dev/aes_ctr_decrypt/benchmark_all.py

# 8. 查看本报告
cat /root/operator-dev/aes_ctr_decrypt/perf_analysis.md
```
