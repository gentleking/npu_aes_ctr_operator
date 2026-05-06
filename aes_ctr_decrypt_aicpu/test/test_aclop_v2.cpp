/*
 * AesCtrDecryptAicpu 算子的单算子调用验证程序
 *
 * 功能:
 *   1. 初始化 AscendCL 运行时
 *   2. 加载 ATC 生成的单算子离线模型 (OM 文件)
 *   3. 准备输入数据（从二进制文件读取密文和密钥）
 *   4. 调用 aclopExecuteV2 执行 AICPU 算子
 *   5. 将输出结果保存到文件
 *
 * 编译方法:
 *   export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
 *   export NPU_HOST_LIB=$DDK_PATH/aarch64-linux/devlib
 *   g++ -std=c++11 test_aclop_v2.cpp \
 *       -I$DDK_PATH/runtime/include \
 *       -L$NPU_HOST_LIB \
 *       -o execute_aes_ctr_decrypt \
 *       -lascendcl
 *
 * 运行方法:
 *   export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:\
 *   /usr/local/Ascend/ascend-toolkit/latest/lib64
 *   export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
 *   ./execute_aes_ctr_decrypt
 *
 * 输出文件: output/result.bin
 */
#include <iostream>    // std::cerr, std::cout
#include <fstream>     // std::ifstream, std::ofstream
#include <vector>      // std::vector
#include <cstdint>     // uint8_t, int64_t

#include "acl/acl.h"   // AscendCL 主头文件

/*
 * main: 验证程序入口
 *
 * 执行流程:
 *   1. aclInit: 初始化 ACL 运行时
 *   2. aclopSetModelDir: 加载 OM 离线模型目录
 *   3. aclrtSetDevice: 绑定 NPU 设备
 *   4. 准备输入/输出数据缓冲区
 *   5. aclopExecuteV2: 执行算子
 *   6. 同步等待执行完成
 *   7. 将结果从 Device 拷贝回 Host
 *   8. 保存结果到文件
 *   9. 释放资源和清理
 */
int main() {
    // ---- 1. 初始化 ACL 运行时 ----
    // scripts/acl.json 可以配置 dump 和 profiling 选项
    // 默认内容为 {} 表示使用默认配置
    if (aclInit("scripts/acl.json") != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return 1;
    }

    // ---- 2. 加载离线模型目录 ----
    // op_models 是 ATC --output 参数指定的目录
    // 目录下的 *.om 文件会被自动加载
    if (aclopSetModelDir("op_models") != ACL_SUCCESS) {
        std::cerr << "aclopSetModelDir failed" << std::endl;
        return 1;
    }

    // ---- 3. 设置 NPU 设备 ----
    // 使用 device 0（可以通过 npu-smi info 查看可用设备）
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl;
        return 1;
    }

    // ---- 4. 检测运行模式 ----
    // ACL_HOST:  Host 侧运行（通过驱动下发任务到 NPU）
    // ACL_DEVICE: Device 侧运行（如 Atals 200DK 上）
    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);
    bool isDevice = (runMode == ACL_DEVICE);

    // ---- 5. 设置数据尺寸 ----
    // 必须与 ATC 转换时指定的 shape 完全一致
    // ATC 时使用: "shape": [1024] for uint8
    int dataSize = 1024;                          // 密文/明文字节数
    std::vector<int64_t> cipherShape = {dataSize}; // 密文 shape: [1024]
    std::vector<int64_t> keyShape = {16};          // 密钥 shape: [16]
    std::vector<int64_t> outShape = {dataSize};    // 明文 shape: [1024]

    // ---- 6. 分配 Device 内存 ----
    void *devCipher, *devKey, *devOut;
    // ACL_MEM_MALLOC_HUGE_FIRST: 优先从大页内存池分配，提高性能
    aclrtMalloc(&devCipher, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devKey, 16, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devOut, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // ---- 7. 读取输入文件 ----
    std::vector<uint8_t> hostCipher(dataSize);
    std::vector<uint8_t> hostKey(16);
    std::vector<uint8_t> hostOut(dataSize);

    // 文件读取 lambda 函数
    auto readFile = [](const char *path, std::vector<uint8_t> &buf) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open " << path << std::endl;
            exit(1);
        }
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    };
    readFile("input/ciphertext.bin", hostCipher);  // 密文
    readFile("input/key.bin", hostKey);             // 密钥

    // ---- 8. 拷贝数据到 Device ----
    aclrtMemcpyKind kind = isDevice ?
        ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE;
    aclrtMemcpy(devCipher, dataSize, hostCipher.data(), dataSize, kind);
    aclrtMemcpy(devKey, 16, hostKey.data(), 16, kind);

    // ---- 9. 创建 Tensor 描述 ----
    // 描述 tensor 的数据类型、维度、格式等信息
    aclTensorDesc *descCipher = aclCreateTensorDesc(
        ACL_UINT8, cipherShape.size(), cipherShape.data(), ACL_FORMAT_ND);
    aclTensorDesc *descKey = aclCreateTensorDesc(
        ACL_UINT8, keyShape.size(), keyShape.data(), ACL_FORMAT_ND);
    aclTensorDesc *descOut = aclCreateTensorDesc(
        ACL_UINT8, outShape.size(), outShape.data(), ACL_FORMAT_ND);

    // ---- 10. 创建数据缓冲区 ----
    // 将 Device 内存地址和数据大小包装为 DataBuffer
    aclDataBuffer *bufCipher = aclCreateDataBuffer(devCipher, dataSize);
    aclDataBuffer *bufKey = aclCreateDataBuffer(devKey, 16);
    aclDataBuffer *bufOut = aclCreateDataBuffer(devOut, dataSize);

    // ---- 11. 创建 Stream ----
    // Stream 用于异步任务管理，同一 stream 内的任务按序执行
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // ---- 12. 构建参数数组并执行算子 ----
    // aclopExecuteV2 取代已废弃的 aclopExecute
    // 参数顺序: opType, numInputs, inputsDesc[], inputs[],
    //           numOutputs, outputsDesc[], outputs[],
    //           attr, stream
    aclTensorDesc *inputsDesc[2] = {descCipher, descKey};
    aclDataBuffer *inputsBuf[2] = {bufCipher, bufKey};
    aclDataBuffer *outputsBuf[1] = {bufOut};
    aclTensorDesc *outputsDesc[1] = {descOut};

    // 执行算子（从已加载的 OM 模型中查找 AesCtrDecryptAicpu 算子执行）
    auto ret = aclopExecuteV2("AesCtrDecryptAicpu",
                              2, inputsDesc, inputsBuf,
                              1, outputsDesc, outputsBuf,
                              nullptr, stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclopExecuteV2 failed: " << ret << std::endl;
        return 1;
    }

    // ---- 13. 同步等待 ----
    // 算子执行是异步的，需等待 stream 中所有任务完成
    aclrtSynchronizeStream(stream);

    // ---- 14. 将结果从 Device 拷贝回 Host ----
    kind = isDevice ?
        ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
    aclrtMemcpy(hostOut.data(), dataSize, devOut, dataSize, kind);

    // ---- 15. 保存输出文件 ----
    std::ofstream fout("output/result.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(hostOut.data()), hostOut.size());
    fout.close();

    std::cout << "Custom op Innovation: Output written to output/result.bin" << std::endl;

    // ---- 16. 资源清理 ----
    // 【重要】必须按相反顺序释放资源，防止内存泄漏
    aclDestroyDataBuffer(bufCipher);
    aclDestroyDataBuffer(bufKey);
    aclDestroyDataBuffer(bufOut);
    aclDestroyTensorDesc(descCipher);
    aclDestroyTensorDesc(descKey);
    aclDestroyTensorDesc(descOut);
    aclrtFree(devCipher);
    aclrtFree(devKey);
    aclrtFree(devOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
