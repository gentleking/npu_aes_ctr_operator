# AES-CTR Operator Development

昇腾 AI 处理器上的 AES-CTR 加解密算子开发项目，包含 AI CPU、AI Core、Ascend C 等多种实现。

## 项目结构

```
operator-dev/
├── aes_ctr_decrypt/             # AI Core (Ascend C) 解密算子 + AclNNInvocation 调用
├── aes_ctr_encrypt/             # AI Core (Ascend C) 加密算子
├── aes_ctr_decrypt_cpu/         # CPU 参考实现（用于验证正确性）
├── aes_ctr_decrypt_aicpu/       # AI CPU 解密算子（本项目的核心）
│   ├── AesCtrDecryptAicpu/      # 算子工程（核函数、原型、编译）
│   ├── test/                    # 测试脚本、ATC 配置、调用程序
│   └── AES_CTR_DECRYPT_AICPU_开发文档.md  # 完整开发文档
├── aes-ctr-custom-v1/           # Ascend C 自定义算子（早期版本）
├── pytorch/                     # PyTorch 适配调用
└── samples/                     # 官方样例
```

## 快速验证（AI CPU 算子）

```bash
cd aes_ctr_decrypt_aicpu/test
bash run.sh
```

一键完成：环境检查 → 生成测试数据 → ATC 转换 OM → 编译 → 执行 → 验证。

## 关键文档

- [完整开发文档](aes_ctr_decrypt_aicpu/AES_CTR_DECRYPT_AICPU_开发文档.md) — 含实现、构建、调用、问题记录

## 环境要求

- 昇腾 AI 处理器（Ascend310P3 等）
- CANN 工具包（8.0.RC 及以上）
- Python 3.7+/3.9+（需 numpy、scipy）
- GCC 7.3+（AICPU 算子需 cross-compiler Do-Compiler GCC 7.3.0）
