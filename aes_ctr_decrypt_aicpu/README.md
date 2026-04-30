# AI-CPU算子开发
## 新建算子工程
```bash
/usr/local/Ascend/ascend-toolkit/8.0.0/python/site-packages/bin/msopgen gen -i AesCtrDecryptAicpu.json -c aicpu  -lan cpp -out AesCtrDecryptAicpu

# 2. 设置 GCC 7 的完整环境
export PATH=/opt/openEuler/gcc-toolset-7/root/usr/bin:$PATH
export CPLUS_INCLUDE_PATH=/opt/openEuler/gcc-toolset-7/root/usr/include/c++/7
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/opt/openEuler/gcc-toolset-7/root/usr/include/c++/7/aarch64-openEuler-linux
export LIBRARY_PATH=/opt/openEuler/gcc-toolset-7/root/usr/lib64:$LIBRARY_PATH


# 3. 设置Python3.7.5环境
export PATH=/usr/local/python3.7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH

```

