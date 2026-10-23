# Ascend 自定义算子 AES-CTR Decrypt 安装指南

## 1. 环境说明

| 项目 | 值 |
|---|---|
| Python | 3.11.14 (`/usr/local/python3.11.14/bin/python3`) |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post1 |
| CANN | 8.5.1 |
| 架构 | aarch64 |
| 算子工程 | `/workspace/ascend-op-dev/aes_ctr_decrypt/` |
| PyTorch 源码 | `/workspace/pytorch/` (Ascend fork, version 2.13.0) |

## 2. 方案对比：C 扩展 vs 修改 torch_npu 源码

| 维度 | 自定义 C 扩展 + torch.library | 修改 torch_npu 源码 + 重编译 |
|---|---|---|
| torch_npu 版本绑定 | 无，自动适配已安装版本 | 需源码版本匹配已安装版本 |
| 构建耗时 | ~1 分钟（仅编译小模块） | ~30 分钟+（完整编译 op-plugin） |
| 调用方式 | `from aes_ctr_decrypt_op import op; op(tensor)` | `torch_npu.npu_aes_ctr_decrypt(tensor)` |
| 对现有环境的影响 | 无，独立模块 | 需 `pip uninstall torch_npu` + 重装 |
| 维护成本 | 低，独立仓库 | 高，每次更新 torch_npu 需重新 patch+rebuild |
| torch.compile 支持 | `torch.library.custom_op` 原生支持 | 需额外注册 |

### 最终选择

**选择：自定义 C 扩展 + torch.library 注册**

理由：
- 当前 torch_npu 2.9.0 已安装且工作正常，不需要重建
- 避免版本匹配问题（Ascend PyTorch 源码为 2.13.0，与已安装的 2.9.0 不匹配）
- 独立模块，不影响原有环境
- 构建速度快

## 3. 安装步骤（已验证通过）

### 3.1 构建 Ascend C 算子并部署到 CANN

```bash
# 安装系统 Python 依赖（算子二进制编译需要）
pip3 install numpy decorator sympy scipy psutil

# 设置 CANN 环境变量
source /usr/local/Ascend/cann-8.5.1/set_env.sh

# 构建算子
cd /workspace/ascend-op-dev/aes_ctr_decrypt/AesCtrDecrypt
rm -rf build_out && bash build.sh

# 安装算子包（第二次生成的 .run 包含 kernel .o）
cd build_out
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}
./custom_opp_openEuler_aarch64.run
```

**验证部署**:
```bash
ls /usr/local/Ascend/cann-8.5.1/opp/vendors/custom_aes_ctr/op_api/lib/libcust_opapi.so
ls /usr/local/Ascend/cann-8.5.1/opp/vendors/custom_aes_ctr/op_impl/ai_core/tbe/kernel/ascend310p/aes_ctr_decrypt/*.o
```

### 3.2 编写 C 扩展 + torch.library 注册

**目录结构**:
```
/workspace/ascend-op-dev/pytorch_extension/aes_ctr_decrypt_op/
├── setup.py                    # setuptools 编译配置
├── __init__.py                 # Python 包 + torch.ops.load_library
├── aes_ctr_decrypt_kernel.cpp  # C++ kernel（调用 aclnnAesCtrDecrypt）
└── test_aes_ctr_decrypt.py     # 完整测试脚本
```

**`aes_ctr_decrypt_kernel.cpp`** 核心流程：
1. 输入 `at::Tensor cyphertext, at::Tensor key`（NPU tensor）
2. 用 `aclCreateTensor(t.sizes().data(), ndim, dtype, nullptr, 0, ACL_FORMAT_ND, sizes.data(), ndim, t.data_ptr())` 创建 aclTensor 包装
3. 调用 `aclnnAesCtrDecryptGetWorkspaceSize` 获取 workspace 大小和 executor
4. 调用 `aclnnAesCtrDecrypt` 执行
5. 同步 stream，返回 `at::Tensor`

**`__init__.py`**（导出方式）：
```python
torch.ops.load_library(path_to_so)
npu_aes_ctr_decrypt = torch.ops.aes_ctr.decrypt
```

### 3.3 编译

```bash
cd /workspace/ascend-op-dev/pytorch_extension/aes_ctr_decrypt_op
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_aes_ctr/op_api/lib/:$LD_LIBRARY_PATH
/usr/local/python3.11.14/bin/python3 setup.py build_ext --build-lib=.
cp aes_ctr_decrypt_op/_C.cpython-311-aarch64-linux-gnu.so .
rm -rf aes_ctr_decrypt_op build
```

### 3.4 测试验证

```bash
cd /workspace/ascend-op-dev/pytorch_extension
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_aes_ctr/op_api/lib/:$LD_LIBRARY_PATH
/usr/local/python3.11.14/bin/python3 aes_ctr_decrypt_op/test_aes_ctr_decrypt.py
```

**测试内容**: CPU 用 pycryptodome 加密 → NPU 用自定义 op 解密 → 断言结果一致

## 4. 回退方案

如果 Ascend C 算子构建失败（环境依赖不足）：

### 备选 A — 使用 AI CPU 算子
- 构建设于 `/workspace/ascend-op-dev/aes_ctr_decrypt_aicpu/`
- 流程类似，kernel 改为 AI CPU 实现，链接和注册方式相同

### 备选 B — 纯 Python 侧解密（无 NPU 算子）
- 绕过 NPU 算子，在 CPU 侧用 `pycryptodome` 完成 AES-CTR 解密
- 无需构建，直接集成到 `model-inference.py`
- 缺点：解密在 CPU 侧，明文暴露在 CPU 内存中
