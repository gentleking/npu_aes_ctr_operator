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
├── benchmark_all.py                  # 统一性能测试脚本（1KB ~ 2GB，覆盖ACLNN和PyTorch两种调用方式）
├── PERF_ANALYSIS.md                 # 性能分析报告 + 复现指南
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
- `BUFFER_NUM` = 2（双缓冲）
- `TILE_LENGTH` = 64（固定tile大小，用于pipeline buffer分配）

**早期版本存在16KB数据量限制**：原始kernel硬编码了`TOTAL_LENGTH = 8192`(half单元，对应16384字节)、`USE_CORE_NUM=8`、`BLOCK_LENGTH=1024`等编译期常量，导致单次调用最多处理16384字节。

**已修复**：从tiling参数`totalLength`动态计算每核分块大小(`thisBlockLen_`)，pipeline迭代次数(`loopCount_`)自适应调整，支持任意数据量（从1字节到2GB均通过验证）。

每个核根据`blockIdx`对计数器做偏移，实现多核独立处理不同数据段。最后一块数据可能不足一个完整tile，由`GetTileLen()`函数处理边界情况。

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

- `src/main.cpp` — 主程序：初始化资源、创建算子描述、设置输入、执行算子、写出输出（已修改为根据输入文件大小动态设置tensor形状）
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
| NPU | Ascend 310P3 |
| PyTorch | 2.1.0 + torch_npu |
| CANN | 8.0.0 |
| Python | 3.9 |
| 算子类型 | AI-CORE (Ascend C, TQue Pipeline, 8 core) |
| 编译模式 | `jit_compile=False`（避免首次编译开销） |

### 基准测试脚本

**统一性能测试脚本** (`benchmark_all.py`)：

覆盖 1KB ~ 2GB 共 17 个数据量级，自动检测可用的调用方式（优先 PyTorch，备选 ACLNN），每轮 warmup 3 次后测量 5 次取统计值，使用 PyCryptodome 验证正确性：

```bash
# PyTorch方式（需torch_npu环境）
/root/miniconda3/envs/operator-dev/bin/python3 benchmark_all.py

# 或使用系统默认python（备选ACLNN方式）
python3 benchmark_all.py
```

### 性能profiling

ACLNN模式下`run.sh`自动调用`msprof op`进行profiling：

```bash
msprof op --output=./benchmark_output execute_add_op
```

PyTorch模式下通过`torch_npu.profiler.profile`采集算子级性能数据，可在Chrome浏览器`chrome://tracing`中打开trace文件进行分析，观察算子执行时间、算子下发调度、Stream同步等细节。

### Kernel 16KB限制修复

**问题**：原始kernel硬编码了`TOTAL_LENGTH = 8192` half单元（=16384字节），所有缓冲区大小和迭代次数均为编译期常量，无法处理大于16KB的数据。

**修复内容**（`op_kernel/aes_ctr_decrypt.cpp`）：
1. 移除编译期常量 `TOTAL_LENGTH`、`USE_CORE_NUM`、`BLOCK_LENGTH`、`TILE_NUM`
2. `Init()`方法改为根据tiling传入的`totalLength`动态计算每核分块大小
3. `Process()`循环次数改为动态计算
4. 新增`GetTileLen()`处理最后不完整tile的边界
5. 新增成员变量`thisBlockLen_`和`loopCount_`

修复后单次kernel调用支持从1字节到2GB任意数据量。

### 大尺寸性能测试结果

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

**关键发现**：
- 耗时与数据量呈完美线性关系，吞吐稳定在 **~77 MB/s**
- 1GB 解密约需 13.3 秒，2GB 约需 26.5 秒
- 单次kernel launch即可处理2GB数据，无kernel launch瓶颈
- 小数据量时（< 1MB），kernel launch启动开销占主导（~80-190us）

### Trace数据分析

`PytorchInvocation/trace/`目录下包含16个Chrome Trace文件，记录了多次profiling结果。

**首次JIT编译开销**：
- 编译耗时 78~618ms（环境差异较大）
- 稳态后kernel仅 ~80-190us
- 编译开销约为稳态的 600~5000x
- `jit_compile=False` 可使首次调用降至 ~0.5ms

**数据搬运开销**：
完整wrapper调用（`custom_aes_ctr_encrypt`）包含 H2D传输 + NPU kernel + D2H传输，其中传输开销约占 85-90%，kernel本身仅占 10-15%。

详细分析见 `PytorchInvocation/trace/PERF_ANALYSIS.md`。

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
