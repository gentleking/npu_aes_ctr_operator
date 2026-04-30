## 目录结构介绍
```
aes_ctr_encrypt
├── AclNNInvocation     // AclNN模式调用单算子
│   ├── README.md
│   ├── inc
│   ├── input
│   ├── run.sh
│   ├── scripts
│   └── src
├── AesCtrEncrypt       // AesCtrEncrypt算子工程化实现
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── CMakePresets.json
│   ├── build.sh
│   ├── framework
│   ├── op_host
│   ├── op_kernel
│   └── scripts
├── AesCtrEncrypt.json  // 算子工程化配置
├── PytorchInvocation   // Pytorch模式调用单算子
│   ├── README.md
│   ├── export_only_prof_dir
│   ├── fusion_result.json
│   ├── op_plugin_patch
│   ├── run_op_plugin.sh
│   ├── benchmark_aes_ctr_encrypt.py
│   ├── test_ops_custom.py
│   ├── test_ops_custom_benchmark.py
│   ├── test_ops_custom_register_in_graph.py
│   └── trace
└── README.md
```

## 算子开发及部署
[算子开发及部署](./AesCtrEncrypt/README.md)

## 单算子执行

### 核模式调用

### AclNNInvocation模式
[AclNNInvocation模式](./AclNNInvocation/README.md)

### PytorchInvocation模式
[PytorchInvocation模式](./PytorchInvocation/README.md)