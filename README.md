# AES-CTR Operator Development

昇腾 AI 处理器上的 AES-CTR 加解密算子开发项目，包含 AI CPU、AI Core、Ascend C 以及 PyTorch C 扩展等多种实现与调用方式。

## 项目结构

```
ascend-op-dev/
├── aes_ctr_decrypt/             # AI Core (Ascend C) 解密算子 + AclNNInvocation 调用
│   ├── AesCtrDecrypt/           # 算子工程（op_host、op_kernel、build.sh）
│   ├── AclNNInvocation/         # ACLNN 直调
│   └── PytorchInvocation/       # PTA 适配与性能测试
├── aes_ctr_encrypt/             # AI Core (Ascend C) 加密算子
├── aes_ctr_decrypt_cpu/         # CPU 参考实现（用于验证正确性）
├── aes_ctr_decrypt_aicpu/       # AI CPU 解密算子
│   ├── AesCtrDecryptAicpu/      # 算子工程（核函数、原型、编译）
│   └── test/                    # 测试脚本、ATC 配置、调用程序
├── aes-ctr-custom-v1/           # Ascend C 自定义算子（早期版本）
├── docs/                        # 安装与开发文档
│   └── ascend_custom_op_install.md
├── pytorch_extension/           # PyTorch C 扩展（torch.library 注册）
│   └── aes_ctr_decrypt_op/
├── tests/                       # 跨模块 / 接口测试
│   └── test_aes_ctr.py
├── packages/                    # 本地归档与参考数据，不随仓库分发
│   ├── archives/                # 算子与环境归档
│   ├── test-assets/             # 构建/测试产物
│   └── references/              # 官方 samples 等参考材料
└── pytorch/                     # torch_npu 源码
```

`packages/` 已加入 `.gitignore`，不会提交到远端；需要分发时使用 Release 附件。

## 快速验证

### AI CPU 算子

```bash
cd aes_ctr_decrypt_aicpu/test
bash run_op_test.sh
```

一键完成：环境检查 → 生成测试数据 → ATC 转换 OM → 编译 → 执行 → 验证。

### AI Core 算子

```bash
cd aes_ctr_decrypt
bash run.sh
```

### PyTorch C 扩展

```bash
cd pytorch_extension/aes_ctr_decrypt_op
source /usr/local/Ascend/cann-8.5.1/set_env.sh
/usr/local/python3.11.14/bin/python3 setup.py build_ext --build-lib=.
```

完整步骤见 [安装指南](docs/ascend_custom_op_install.md)。

### 接口测试

```bash
cd tests
/usr/local/python3.11.14/bin/python3 test_aes_ctr.py
```

`tests/test_aes_ctr.py` 依赖 `/workspace/LMCache-Ascend` 中的 `aes_ctr_encrypt` 封装。

## 关键文档

- [安装指南](docs/ascend_custom_op_install.md) — Ascend C 算子构建、CANN 部署、PyTorch C 扩展编译与测试
- [AI CPU 开发文档](aes_ctr_decrypt_aicpu/AES_CTR_DECRYPT_AICPU_开发文档.md) — 实现、构建、调用、问题记录
- [算子开发指南](aes_ctr_decrypt/AesCtrDecrypt/op_dev_guide.md) — Tiling、流水线、数据类型与优化方向
- [AI Core 性能分析](aes_ctr_decrypt/perf_analysis.md)
- [PyTorch 调用性能分析](aes_ctr_decrypt/PytorchInvocation/PERF_ANALYSIS.md)

## 环境要求

- 昇腾 AI 处理器（Ascend310P3 等）
- CANN 工具包（8.0.RC 及以上）
- Python 3.7+/3.9+（需 numpy、scipy）
- GCC 7.3+（AICPU 算子需 cross-compiler Do-Compiler GCC 7.3.0）
