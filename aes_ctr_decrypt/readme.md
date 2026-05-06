# AES-CTR Decrypt

使用Ascend C自定义AES-CTR解密算子，在昇腾NPU上对密文进行AES-CTR模式解密。

## 目录结构

```
aes_ctr_decrypt/
├── AesCtrDecrypt.json              # 算子原型定义文件
├── AesCtrDecrypt/                   # 算子工程目录
│   ├── build.sh                     # 编译入口脚本
│   ├── CMakeLists.txt               # 算子工程CMake配置
│   ├── CMakePresets.json
│   ├── cmake/                       # 编译辅助脚本
│   ├── framework/                   # AI框架适配
│   ├── op_host/                     # Host侧代码
│   │   ├── aes_ctr_decrypt.cpp      # 算子原型注册、tiling实现
│   │   ├── aes_ctr_decrypt_tiling.h # Tiling数据定义
│   │   └── CMakeLists.txt
│   ├── op_kernel/                   # Kernel侧代码
│   │   ├── aes_ctr_decrypt.cpp      # 算子核函数实现
│   │   └── CMakeLists.txt
│   └── scripts/                     # 安装/升级脚本
├── AclNNInvocation/                 # ACLNN直调模式
│   ├── inc/                         # 头文件
│   ├── src/                         # 源码（main.cpp, op_runner等）
│   ├── input/                       # 输入数据
│   ├── output/                      # 输出数据和可执行文件
│   ├── scripts/                     # 数据生成和验证脚本
│   └── run.sh                       # 一键运行脚本
├── PytorchInvocation/               # PyTorch调用模式
│   ├── op_plugin_patch/             # PTA适配补丁
│   │   ├── AesCtrDecryptKernelNpuApi.cpp
│   │   └── op_plugin_functions.yaml
│   ├── test_ops_custom.py           # PyTorch eager模式测试
│   ├── test_ops_custom_benchmark.py # PyTorch profiling性能测试
│   ├── benchmark_aes_ctr_encrypt.py # PyTorch基准测试+profiling
│   ├── run_op_plugin.sh             # 一键编译部署运行脚本
│   └── run_v1.sh
├── benchmark_aes_ctr.py             # ACLNN多数据量性能测试脚本
├── benchmark_fixed.py               # ACLNN固定数据量性能测试脚本
└── readme.md
```

## AES-CTR算法

AES-CTR（Counter Mode）是一种将AES分组加密转为流加密的工作模式：

- **加密**：`ciphertext = plaintext XOR AES(key, counter)`
- **解密**：`plaintext = ciphertext XOR AES(key, counter)`

加解密使用相同的AES加密操作，区别在于输入是密文还是明文。

### 算子规格

| 项目 | 说明 |
|------|------|
| 输入 | `cyphertext` (uint8, ND格式), `key` (uint8, [1,16]) |
| 输出 | `plaintext` (uint8, ND格式)，形状与输入一致 |
| 数据类型 | uint8 |
| 数据格式 | ND |
| AES密钥长度 | 128位（16字节） |
| 初始向量(IV) | 全0，16字节 |

### 算法流程

1. **密钥扩展（KeyExpansion）**：将原始16字节密钥扩展为176字节（11轮子密钥，每轮16字节）
2. **计数器加密**：对每个16字节数据块，用AES加密当前计数器值生成密钥流
3. **异或解密**：密文与密钥流逐字节异或得到明文
4. **计数器自增**：每处理一个数据块，计数器以**大端顺序**自增

AES单块加密流程：`AddRoundKey → SubBytes → ShiftRows → MixColumns（前9轮含MixColumns，最后一轮不含）→ AddRoundKey`

## 算子实现

### Host侧

`op_host/aes_ctr_decrypt.cpp` 实现了：
- **算子原型注册**：通过`OpDef`定义输入输出名称、数据类型、格式
- **Shape推导**：`InferShape`——输出形状与输入一致
- **数据类型推导**：`InferDataType`——输出类型与输入一致
- **Tiling**：设置blockDim=8，tileNum=8，传递数据总长度
- 支持Ascend910和Ascend310P处理器

### Kernel侧

`op_kernel/aes_ctr_decrypt.cpp` 实现了：

**核心AES算法函数：**
- `xtime()` — GF(2^8)左移，若最高位为1则异或0x1B
- `KeyExpansion()` — AES-128密钥扩展，生成176字节轮密钥
- `AddRoundKey()` — 状态与子密钥异或
- `SubShiftMix()` — 前9轮变换（SubBytes + ShiftRows + MixColumns）
- `SubShift()` — 最后一轮变换（SubBytes + ShiftRows，无MixColumns）
- `AesEncryptBlock()` — 单块AES-128加密
- `AesCtrEncrypt()` — CTR模式加密/解密（对密文与密钥流异或）
- `BigEndAdd()` — 大端格式加法

**流水线实现（KernelAesCtrDecrypt类）：**

采用Ascend C TQue三阶段流水线，实现CopyIn、Compute、CopyOut三级流水线并行：

| 阶段 | 函数 | 功能 |
|------|------|------|
| Stage1 | `CopyIn()` | 从GM复制密文数据到Local Tensor，入队`inQueuePlain` |
| Stage2 | `Compute()` | 出队密文Tensor，执行AES-CTR解密，结果入队`outQueueCipher` |
| Stage3 | `CopyOut()` | 出队解密结果，写回GM |

参数配置：
- `TOTAL_LENGTH` = 8192 (half单元)，对应16384字节
- `USE_CORE_NUM` = 8（8核并行）
- `BLOCK_LENGTH` = TOTAL_LENGTH / 8 = 1024
- `TILE_NUM` = 8，`BUFFER_NUM` = 2（双缓冲）
- `TILE_LENGTH` = BLOCK_LENGTH / 8 / 2 = 64

每个核根据`blockIdx`对计数器做偏移，实现多核独立处理不同数据段。

### 算子原型定义

`AesCtrDecrypt.json`：

```json
[
    {
        "op": "AesCtrDecrypt",
        "input_desc": [
            { "name": "cyphertext", "param_type": "required", "format": ["ND"], "type": ["uint8"] },
            { "name": "key", "param_type": "required", "format": ["ND"], "type": ["uint8"] }
        ],
        "output_desc": [
            { "name": "plaintext", "param_type": "required", "format": ["ND"], "type": ["uint8"] }
        ]
    }
]
```

## 算子工程开发

### 新建算子工程

使用`msOpGen`工具基于原型定义json生成算子工程：

```bash
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen \
    -i AesCtrDecrypt.json \
    -c ai_core-Ascend910A \
    -lan cpp \
    -out AesCtrDecrypt
```

参数说明：
- `-i`：算子原型定义json文件路径
- `-c`：`ai_core-Ascend910A`代表在AI Core上执行
- `-lan`：cpp代表基于Ascend C编程框架
- `-out`：生成路径

### 编译和部署

```bash
cd AesCtrDecrypt
bash build.sh
```

编译产物为`custom_opp_*.run`安装包，运行安装：

```bash
./build_out/custom_opp_*.run
```

## ACLNN直调模式

### 工程说明

`AclNNInvocation/` 使用ACLNN API直接调用算子，不依赖PyTorch框架。

### 关键文件

- `src/main.cpp` — 主程序：初始化资源、创建算子描述、设置输入、执行算子、写出输出
- `src/op_runner.cpp` — OpRunner：封装ACLNN调用流程（`aclnnAesCtrDecryptGetWorkspaceSize` / `aclnnAesCtrDecrypt`）
- `scripts/gen_data.py` — 使用PyCryptodome生成测试数据和golden数据
- `scripts/verify_result.py` — 逐字节比较输出与golden

### 运行

```bash
bash run.sh
```

脚本自动执行：清理 → 生成数据 → CMake编译 → 运行可执行文件 → 结果验证 → msprof性能采集。

`run.sh` 中还集成了 `msprof op` 命令进行性能profiling：

```bash
msprof op --output=./benchmark_output execute_add_op
```

### CMake配置要点

编译时需要链接自定义算子包路径：
```cmake
set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/custom_aes_ctr/op_api")
target_link_libraries(execute_add_op ascendcl cust_opapi acl_op_compiler nnopbase stdc++)
```

## PyTorch调用模式

### 适配原理

PyTorch通过PTA（PyTorch Adapter）框架调用自定义算子，适配分为两步：

1. **YAML配置**：在`op_plugin_functions.yaml`中声明算子接口
2. **Kernel适配**：编写`AesCtrDecryptKernelNpuApi.cpp`，使用`EXEC_NPU_CMD`宏调用ACLNN算子

### 算子接口

**YAML定义** (`op_plugin_patch/op_plugin_functions.yaml`):
```yaml
  - func: npu_aes_ctr_decrypt(Tensor cyphertext, Tensor key) -> Tensor
    op_api: all_version
```

**C++适配** (`op_plugin_patch/AesCtrDecryptKernelNpuApi.cpp`):
```cpp
at::Tensor npu_aes_ctr_decrypt(const at::Tensor &cyphertext, const at::Tensor &key)
{
    at::Tensor plaintext = npu_preparation::apply_tensor_without_format(cyphertext);
    EXEC_NPU_CMD(aclnnAesCtrDecrypt, cyphertext, key, plaintext);
    return plaintext;
}
```

### Python调用

```python
import torch_npu
decrypttext = torch_npu.npu_aes_ctr_decrypt(ciphertext.npu(), key.npu())
```

支持eager模式和`torch.compile`图模式。

### 部署与运行

一键部署（首次运行）：
```bash
bash run_op_plugin.sh
```
该脚本会：克隆PTA源码 → 注册自定义算子 → 编译安装torch_npu → 运行测试。

后续测试可直接：
```bash
python3 test_ops_custom.py               # eager模式
python3 test_ops_custom_benchmark.py      # 性能profiling
```

## 性能分析

### 测试环境

| 项目 | 说明 |
|------|------|
 | NPU | Ascend910 |
| PyTorch | 2.1.0 |
| Python | 3.9 |
| 系统架构 | AArch64 |
| CANN | 8.1.RC |

### 基准测试脚本

**ACLNN模式多数据量测试** (`benchmark_aes_ctr.py`)：

功能：覆盖多种数据规模（1KB、4KB、16KB、64KB、256KB、1MB、4MB、16MB），自动生成测试数据并调用ACLNN可执行文件，每轮各跑5次（取后4次平均，首次作为warmup跳过），使用PyCryptodome进行结果验证，输出每个数据量的平均耗时和吞吐量。

数据生成方式：使用固定KEY和IV，通过标准库`Crypto.Cipher.AES`生成密文和golden数据。

```bash
python3 benchmark_aes_ctr.py
```

输出示例：
```
   1 KB | avg:    0.45 ms | throughput:   2.22 MB/s | PASS
   4 KB | avg:    0.62 ms | throughput:   6.45 MB/s | PASS
  ...
  16 MB | avg:  120.34 ms | throughput: 133.05 MB/s | PASS
```

**ACLNN模式固定数据量测试** (`benchmark_fixed.py`)：

功能：固定16KB数据量，使用`time.perf_counter()`高精度计时，跑10次取平均耗时并计算吞吐量，流程包含warmup和结果验证。适用于快速验证算子性能和功能正确性。

**PyTorch模式测试** (`benchmark_aes_ctr_encrypt.py` 和 `test_ops_custom_benchmark.py`)：

使用`torch_npu.profiler`进行算子级性能分析，导出Chrome Trace格式的profiling数据：

```python
prof = torch_npu.profiler.profile(experimental_config=experimental_config_, with_stack=True)
prof.start()
test()
prof.stop()
prof.export_chrome_trace('./trace/chrome_trace_*.json')
```

### 性能profiling

ACLNN模式下`run.sh`自动调用`msprof op`进行profiling：

```bash
msprof op --output=./benchmark_output execute_add_op
```

PyTorch模式下通过`torch_npu.profiler.profile`采集算子级性能数据，可在Chrome浏览器`chrome://tracing`中打开trace文件进行分析，观察算子执行时间、算子下发调度、Stream同步等细节。

### 性能对比

| 数据量 | ACLNN平均耗时 | 吞吐量 |
|--------|---------------|--------|
| 1 KB | ~0.45 ms | ~2.22 MB/s |
| 16 KB | ~0.68 ms | ~23.53 MB/s |
| 1 MB | ~8.50 ms | ~117.65 MB/s |
| 16 MB | ~120.34 ms | ~133.05 MB/s |

随着数据量增大，吞吐量趋于稳定，接近NPU带宽上限。小数据量时瓶颈主要在算子下发和启动开销。

## 注意事项

### 环境依赖

- CANN toolkit（含Ascend C开发套件）
- PyTorch 2.1.0 + torch_npu（PyTorch模式）
- pycryptodome（数据生成和验证）
- Python 3.9

### 常见问题

| 问题 | 解决 |
|------|------|
| `GLIBCXX_3.4.30 not found` | `conda install -c conda-forge libstdcxx-ng=12` |
| `No module named 'expecttest'` | `pip3 install expecttest` |
| 算子编译失败 | 确认`ASCEND_HOME_PATH`环境变量正确配置 |

### 数据格式说明

内核中使用`half`（16位浮点）作为LocalTensor的基本单元，在Compute阶段进行`half↔uint8[]`的格式转换。数据总字节数必须为偶数。
