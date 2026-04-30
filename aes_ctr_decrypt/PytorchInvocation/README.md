# PytorchInvocation调用
## 目录结构
```
.
├── README.md
├── benchmark_aes_ctr_encrypt.py
├── export_only_prof_dir
├── fusion_result.json
├── op_plugin_patch
│   ├── AesCtrEncryptKernelNpuApi.cpp
│   └── op_plugin_functions.yaml
├── pytorch
├── run_op_plugin.sh
├── test_ops_custom.py
├── test_ops_custom_benchmark.py
├── test_ops_custom_register_in_graph.py
└── trace
```


## 编译和部署算子
运行此样例前，请完成前期准备工作，包括算子开发、编译和部署。

## 调用算子
本项目Pytorch=2.1.0，Python=3.9，系统架构=AArch64，CANN=8.1.RC

### Pytorch调用算子

  - 样例执行
    1) 第一次运行时使用脚本运行
    样例执行过程中会自动生成测试数据，然后运行pytorch样例，最后检验运行结果。具体过程可参见run_op_plugin.sh脚本。
    ```bash
    bash run_op_plugin.sh
    ```

    2) 后续测试可以单独执行python脚本，避免重复安装torch_npu
    ```bash
    python3 test_ops_custom.py    // eager模式
    python3 test_ops_custom_register_in_graph.py
    ```

### 算子性能测试
代码样例
```python
    experimental_config_ = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2
    )
    
    prof = torch_npu.profiler.profile(
        experimental_config=experimental_config_,
        with_stack=True,
    )
    
    prof.start()
    test()
    prof.stop()
    
    os.makedirs('./trace', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filepath = f'./trace/chrome_trace_{timestamp}.json'
    prof.export_chrome_trace(filepath)
    print(f"Profile trace exported to {filepath}")
```

## 遇到的问题

- 安装pycryptodome
```bash
pip3 install pycryptodome
```

- 安装torch，参考[torch安装](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)
```bash
# 下载软件包
wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 安装命令
pip3 install torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

- No module named 'expecttest'
```bash
ModuleNotFoundError: No module named 'expecttest'
```
解决办法
```bash
pip3 install expecttest
```

- GLIBCXX_3.4.30' not found
```bash
ImportError: /root/miniconda3/envs/operator-dev/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /root/miniconda3/envs/operator-dev/lib/python3.9/site-packages/torch_npu/lib/libtorch_npu.so)
```
解决办法
```bash
conda install -c conda-forge libstdcxx-ng=12
```

