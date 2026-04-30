## 算子工程化、开发及部署

### 环境配置
```bash
pip3 install attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py
```

### 算子工程化
- 编辑算子工程化配置文件
例：AesCtrEncrypt.json
```json
[
    {
        "op": "AesCtrEncrypt",
        "input_desc": [
            {
                "name": "plaintext",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "uint8"
                ]
            },
            {
                "name": "key",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "uint8"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "cyphertext",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "uint8"
                ]
            }
        ]
    }
]

```

- 自动生成工程化算子
使用工具msopgen工具根据配置文件自动生成算子工程
例：
```bash
/usr/local/Ascend/ascend-toolkit/8.0.0/python/site-packages/bin/msopgen gen -i AesCtrEncrypt.json -c ai_core-Ascend310P3 -lan cpp -out AesCtrEncrypt_ascend310p3
```

### 算子工程开发

修改**op_host/aes_ctr_encrypt_tiling.h**、**op_host/aes_ctr_encrypt.cpp**和**op_kernel/aes_ctr_encrpt.cpp**文件

### 算子工程编译
编译配置文件为CMakePresets.json，运行命令
```bash
./build.sh
```

### 算子工程部署
```bash
./build_out/custom_opp_openEuler_aarch64.run