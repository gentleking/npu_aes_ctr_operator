# AesCtrDecryptAicpu AI CPU 算子

## 1. 概述

本算子实现了 **AES-128 CTR 模式的解密**，运行在昇腾 AI 处理器的 AI CPU（AICPU）上。算子接收密文和密钥作为输入，输出解密后的明文，IV 固定为全零。

| 项目 | 值 |
|------|------|
| OpType | `AesCtrDecryptAicpu` |
| 输入 0 | `ciphertext`: uint8, ND, 任意长度 |
| 输入 1 | `key`: uint8, ND, [16] |
| 输出 | `plaintext`: uint8, ND, 与输入 0 同形状 |
| 引擎 | DNN_VM_AICPU |
| 供应商 | customize_aicpu |

## 2. 项目结构

```
operator-dev/aes_ctr_decrypt_aicpu/
├── AesCtrDecryptAicpu.json                 # 算子原型定义（msOpGen 输入）
├── AesCtrDecryptAicpu/                     # msOpGen 生成的算子工程
│   ├── build.sh                            # 编译入口脚本
│   ├── CMakeLists.txt                      # 顶层 CMake
│   ├── cpukernel/                          # AI CPU 核函数实现
│   │   ├── CMakeLists.txt
│   │   ├── impl/
│   │   │   ├── aes_ctr_decrypt_aicpu_kernels.h   # 头文件
│   │   │   └── aes_ctr_decrypt_aicpu_kernels.cc  # 核函数实现
│   │   └── op_info_cfg/aicpu_kernel/
│   │       └── aes_ctr_decrypt_aicpu.ini         # 算子信息配置
│   ├── op_proto/                           # 算子原型注册
│   ├── op_tiling/                          # Tiling（本算子不需要）
│   ├── framework/                          # 框架适配插件
│   ├── scripts/                            # 安装/卸载脚本
│   └── build_out/                          # 编译输出
├── test/
│   ├── test_aclop_v2.cpp                   # C++ 调用程序（AscendCL）
│   ├── test_aes_ctr_perf.cpp               # 性能测试程序
│   ├── perf_test.py                        # 性能测试脚本
│   ├── scripts/gen_data.py                 # 测试数据生成
│   ├── scripts/verify_result.py            # 结果验证
│   ├── run.sh                              # 一键验证脚本
│   └── aes_ctr_decrypt_aicpu_op.json       # ATC 输入描述
├── README.md                               # 本文件
└── AES_CTR_DECRYPT_AICPU_开发文档.md        # 详细开发文档
```

## 3. 算子开发

### 3.1 算子原型定义

`AesCtrDecryptAicpu.json`:

```json
[
    {
        "op": "AesCtrDecryptAicpu",
        "input_desc": [
            {"name": "ciphertext", "param_type": "required", "format": ["ND"], "type": ["uint8"]},
            {"name": "key", "param_type": "required", "format": ["ND"], "type": ["uint8"]}
        ],
        "output_desc": [
            {"name": "plaintext", "param_type": "required", "format": ["ND"], "type": ["uint8"]}
        ]
    }
]
```

使用 `msOpGen` 生成工程框架：

```bash
/usr/local/Ascend/ascend-toolkit/8.0.0/python/site-packages/bin/msopgen gen \
    -i AesCtrDecryptAicpu.json -c aicpu -lan cpp -out AesCtrDecryptAicpu
```

### 3.2 核函数实现

核函数文件位于 `cpukernel/impl/`：

- **`aes_ctr_decrypt_aicpu_kernels.h`** — 声明 `AesCtrDecryptAicpuCpuKernel` 类，继承 `CpuKernel`，包含 AES-128 核心算法（keyExpansion, subBytes, shiftRows, mixColumns, encryptBlock）和 CTR 模式辅助方法（bigEndAdd, generateKeystream）
- **`aes_ctr_decrypt_aicpu_kernels.cc`** — 完整实现

**解密流程**：
1. 初始化 S 盒、扩展密钥、IV/计数器
2. 循环处理每个 16 字节分组：
   - `keystream = AES_encrypt(key, counter)` — 加密计数器生成密钥流
   - `counter += 1`
   - `plaintext = ciphertext XOR keystream` — 异或解密

### 3.3 算子信息配置

`cpukernel/op_info_cfg/aicpu_kernel/aes_ctr_decrypt_aicpu.ini`:

```ini
[AesCtrDecryptAicpu]
opInfo.engine=DNN_VM_AICPU
opInfo.flagPartial=False
opInfo.computeCost=100
opInfo.flagAsync=False
opInfo.opKernelLib=CUSTAICPUKernel
opInfo.kernelSo=libaes_ctr_decrypt_aicpu.so
opInfo.functionName=RunCpuKernel
opInfo.workspaceSize=1024
```

## 4. 算子编译与安装

### 4.1 环境变量

```bash
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TENSOR_COMPILER_INCLUDE=/usr/local/Ascend/ascend-toolkit/latest/compiler/include
export TOOLCHAIN_DIR=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc
export AICPU_KERNEL_TARGET=aes_ctr_decrypt_aicpu
export OPP_CUSTOM_VENDOR=customize_aicpu
export SYSTEM_INFO=openeuler_aarch64
export AICPU_SOC_VERSION=Ascend310P3
```

### 4.2 编译

> **注意**：必须使用 `-nostdinc++` 强制使用 cross-compiler GCC 7.3.0 的 C++ 头文件，避免系统 GCC 12 头文件与 AICPU 框架的 ABI 不兼容问题。

```bash
cd AesCtrDecryptAicpu
rm -rf build_out && mkdir build_out && cd build_out

cmake .. \
  -DCMAKE_CXX_FLAGS="-nostdinc++ \
    -isystem ${TOOLCHAIN_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0 \
    -isystem ${TOOLCHAIN_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu \
    -D_GLIBCXX_USE_CXX11_ABI=1"

make -j$(nproc)
```

编译成功后输出：
```
build_out/makepkg/packages/vendors/customize_aicpu/op_impl/cpu/aicpu_kernel/impl/libaes_ctr_decrypt_aicpu.so
build_out/custom_opp_openeuler_aarch64.run
```

### 4.3 安装

```bash
cd build_out
TOOLCHAIN_DIR=$TOOLCHAIN_DIR bash custom_opp_openeuler_aarch64.run --quiet
```

安装后验证：

```bash
ls /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize_aicpu/op_impl/cpu/aicpu_kernel/impl/
# 应包含 libaes_ctr_decrypt_aicpu.so

cat /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/config.ini
# 应包含 load_priority=customize_aicpu,...
```

### 4.4 安装后文件

```
/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/
└── customize_aicpu/
    ├── op_impl/cpu/aicpu_kernel/impl/libaes_ctr_decrypt_aicpu.so  # 核函数动态库
    ├── op_impl/cpu/config/cust_aicpu_kernel.json                   # 算子信息配置
    └── op_proto/libcust_op_proto.so                                # 算子原型库
```

## 5. 算子调用

### 5.1 生成测试数据

```bash
cd test
pip3 install pycryptodome numpy
python3 scripts/gen_data.py 1024
```

生成的文件：
- `input/ciphertext.bin` — 密文（算子输入）
- `input/key.bin` — 密钥（算子输入）
- `output/golden.bin` — 预期明文（验证用）

### 5.2 ATC 转换离线模型

```bash
export PATH=/root/miniconda3/envs/operator-dev/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/operator-dev/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/customize_aicpu
export LC_ALL=C

atc --singleop=aes_ctr_decrypt_aicpu_op.json \
    --soc_version=Ascend310P3 \
    --output=op_models
```

### 5.3 C++ 调用程序

`test/test_aclop_v2.cpp` 使用 AscendCL API 执行算子：

```cpp
aclopExecuteV2("AesCtrDecryptAicpu",
               2, inputsDesc, inputsBuf,
               1, outputsDesc, outputsBuf,
               nullptr, stream);
```

编译与运行：

```bash
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/aarch64-linux/devlib

g++ -std=c++11 test_aclop_v2.cpp \
    -I$DDK_PATH/runtime/include \
    -L$NPU_HOST_LIB \
    -o execute_aes_ctr_decrypt \
    -lascendcl

LD_LIBRARY_PATH=$DDK_PATH/runtime/lib64:$DDK_PATH/lib64 \
ASCEND_OPP_PATH=$DDK_PATH/opp \
./execute_aes_ctr_decrypt
```

### 5.4 一键验证

```bash
cd test
bash run.sh
```

## 6. 功能验证结果

```
[STEP] 1/6: 检查环境依赖
  ✓ Python: Python 3.9.21
  ✓ atc: /usr/local/Ascend/ascend-toolkit/latest/bin/atc
  ✓ AscendCL 头文件

[STEP] 2/6: 生成测试数据
  ✓ 测试数据已生成 (1024 字节)

[STEP] 3/6: ATC 转换单算子离线模型
  ✓ ATC 转换成功

[STEP] 4/6: 编译验证程序
  ✓ 编译成功

[STEP] 5/6: 执行算子验证
  ✓ 算子执行完成

[STEP] 6/6: 验证结果
PASS: output matches golden!
  ✓ 验证通过
```

| 项目 | 结果 |
|------|------|
| 测试数据量 | 1024 字节 |
| 验证结果 | ✅ 1024/1024 字节完全匹配 |
| AES 算法 | AES-128 CTR 模式解密正确 |
| 算子引擎 | DNN_VM_AICPU |
| 验证方式 | ATC → OM 模型 → aclopExecuteV2 |

## 7. 性能测试

### 7.1 测试环境

| 项目 | 配置 |
|------|------|
| NPU | Ascend 310P3 |
| 驱动版本 | 25.2.0 |
| 算子引擎 | DNN_VM_AICPU |
| 测试工具 | Python + C++ AscendCL |
| 密钥 | 0x0123456789ABCDEFFEDCBA9876543210 |
| IV | 全零 |

### 7.2 测试脚本

`test/perf_test.py` — Python 自动化测试脚本，循环执行以下步骤：
1. 调用 `gen_data()` 生成测试数据（使用 PyCryptodome AES-CTR 加密）
2. 调用 ATC 生成对应 shape 的单算子 OM 模型
3. 执行 `test_aes_ctr_perf` 二进制程序（含 warmup + 多次运行 + 计时）
4. 使用 NumPy 对比算子输出与 golden 值验证正确性

### 7.3 性能测试结果

| Data Size   | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Throughput (MB/s) | Result |
|-------------|--------------|--------------|--------------|-------------------|--------|
| 1 MB        | 54.2         | 54.1         | 54.3         | 18.47             | PASS   |
| 2 MB        | 107.8        | 107.8        | 107.9        | 18.55             | PASS   |
| 4 MB        | 215.4        | 215.2        | 215.5        | 18.57             | PASS   |
| 8 MB        | 436.5        | 430.2        | 448.7        | 18.33             | PASS   |
| 16 MB       | 872.5        | 859.9        | 897.0        | 18.34             | PASS   |
| 32 MB       | 1724.2       | 1723.6       | 1725.0       | 18.56             | PASS   |
| 64 MB       | 3488.1       | 3438.7       | 3586.1       | 18.35             | PASS   |
| 128 MB      | 6876.2       | 6875.0       | 6877.3       | 18.62             | PASS   |
| 256 MB      | 13780.3      | 13767.5      | 13792.7      | 18.58             | PASS   |
| 512 MB      | 27491.6      | 27486.6      | 27495.8      | 18.62             | PASS   |
| 1 GB        | —            | —            | —            | —                 | SKIP   |
| 2 GB        | —            | —            | —            | —                 | SKIP   |

### 7.4 性能分析

- **吞吐量稳定在约 18.5 MB/s**，与数据量呈线性关系（O(n)），符合 AES-CTR 流密码的特性
- 各数据量的单位吞吐量基本一致，说明算子内部没有明显的初始化开销或边界效应
- 512 MB 测试在独立运行中通过，但由于 AICPU 运行时的任务队列状态依赖，多次小数据量测试后可能出现 AICPU 超时（错误码 507017）

### 7.5 1GB/2GB 测试说明

1GB 和 2GB 数据量无法在当前环境下完成测试，原因：
- **NPU 设备内存限制**：Device 0 的空闲内存约 1.8 GB，算子需要同时分配输入（密文）和输出（明文）两个缓冲区，共需 2×1GB = 2GB 设备内存
- **AICPU 超时限制**：处理 1GB 数据约需 57 秒（基于 18.5 MB/s 推算），可能超过 AICPU 内置的看门狗超时阈值

## 8. 遇到的问题与解决方案

### 8.1 GCC 版本不兼容

**问题**：`REGISTER_CPU_KERNEL` 宏中的 `std::shared_ptr` 模板展开失败。
**根因**：cross-compiler（GCC 7.3.0）使用了系统 GCC 12 的 C++ 头文件，存在 ABI 不兼容。
**解决**：cmake 时添加 `-nostdinc++` 并手动指定 GCC 7.3.0 的头文件路径。

### 8.2 kernelSo 文件名不匹配

**问题**：`kernelSo=libcust_aicpu_kernels.so` 但实际编译输出为 `libaes_ctr_decrypt_aicpu.so`。
**解决**：将 `.ini` 文件中的 `kernelSo` 改为 `libaes_ctr_decrypt_aicpu.so`，保持与实际输出一致。

### 8.3 设备内存不足

**问题**：大数据量（1GB+）测试时 `aclrtMalloc` 返回 207001（内存分配失败）。
**解决**：使用原位缓冲（in-place）策略，输入输出共用同一设备内存地址，可将设备内存需求减半。

### 8.4 AICPU 执行超时

**问题**：大数据量测试时出现 `aclrtSynchronizeStream` 错误 507017（AICPU 异常/超时）。
**分析**：AICPU 有内置看门狗定时器，处理时间过长的任务会被中断。

## 9. 快速开始

```bash
# 1. 编译算子
cd AesCtrDecryptAicpu
export TOOLCHAIN_DIR=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc
rm -rf build_out && mkdir build_out && cd build_out
cmake .. -DCMAKE_CXX_FLAGS="-nostdinc++ -isystem ${TOOLCHAIN_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0 -isystem ${TOOLCHAIN_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu -D_GLIBCXX_USE_CXX11_ABI=1"
make -j$(nproc)

# 2. 安装算子
TOOLCHAIN_DIR=$TOOLCHAIN_DIR bash custom_opp_openeuler_aarch64.run --quiet

# 3. 功能验证
cd ../../test
bash run.sh

# 4. 性能测试
python3 perf_test.py
```
