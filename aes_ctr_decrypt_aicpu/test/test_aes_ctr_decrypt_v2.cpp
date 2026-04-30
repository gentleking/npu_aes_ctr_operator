#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"

const int DEVICE_ID = 0;

int main(int argc, char **argv) {
    const char *opModelDir = (argc > 1) ? argv[1] : "op_models";
    const char *cipherFile = (argc > 2) ? argv[2] : "input/ciphertext.bin";
    const char *keyFile = (argc > 3) ? argv[3] : "input/key.bin";
    const char *outputFile = (argc > 4) ? argv[4] : "output/result.bin";
    int dataSize = (argc > 5) ? std::stoi(argv[5]) : 1024;

    if (aclInit("scripts/acl.json") != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl; return 1;
    }
    if (aclopSetModelDir(opModelDir) != ACL_SUCCESS) {
        std::cerr << "aclopSetModelDir failed" << std::endl; return 1;
    }
    if (aclrtSetDevice(DEVICE_ID) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl; return 1;
    }

    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);
    bool isDevice = (runMode == ACL_DEVICE);

    std::vector<int64_t> cipherShape = {dataSize};
    std::vector<int64_t> keyShape = {16};
    std::vector<int64_t> outShape = {dataSize};

    size_t cipherBytes = dataSize;
    size_t keyBytes = 16;
    size_t outBytes = dataSize;

    void *devCipher, *devKey, *devOut;
    aclrtMalloc(&devCipher, cipherBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devKey, keyBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    std::vector<uint8_t> hostCipher(cipherBytes), hostKey(keyBytes), hostOut(outBytes, 0xAA);

    auto readFile = [](const char *path, std::vector<uint8_t> &buf) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << path << std::endl; exit(1); }
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    };
    readFile(cipherFile, hostCipher);
    readFile(keyFile, hostKey);

    aclrtMemcpyKind kind = isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE;
    aclrtMemcpy(devCipher, cipherBytes, hostCipher.data(), cipherBytes, kind);
    aclrtMemcpy(devKey, keyBytes, hostKey.data(), keyBytes, kind);

    aclTensorDesc *descCipher = aclCreateTensorDesc(ACL_UINT8, cipherShape.size(), cipherShape.data(), ACL_FORMAT_ND);
    aclTensorDesc *descKey = aclCreateTensorDesc(ACL_UINT8, keyShape.size(), keyShape.data(), ACL_FORMAT_ND);
    aclTensorDesc *descOut = aclCreateTensorDesc(ACL_UINT8, outShape.size(), outShape.data(), ACL_FORMAT_ND);

    aclDataBuffer *bufCipher = aclCreateDataBuffer(devCipher, cipherBytes);
    aclDataBuffer *bufKey = aclCreateDataBuffer(devKey, keyBytes);
    aclDataBuffer *bufOut = aclCreateDataBuffer(devOut, outBytes);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    // Use aclopExecuteV2
    size_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const char *opType = "AesCtrDecryptAicpu";

    auto ret = aclopCompileAndExecute(opType, 2,
        &descCipher, &descKey,
        1, &descOut,
        2, &bufCipher, &bufKey,
        1, &bufOut,
        nullptr, nullptr,
        0, nullptr,
        stream);

    if (ret != ACL_SUCCESS) {
        std::cerr << "aclopCompileAndExecute failed: " << ret << std::endl;
        return 1;
    }

    aclrtSynchronizeStream(stream);

    kind = isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
    aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes, kind);

    std::ofstream fout(outputFile, std::ios::binary);
    fout.write(reinterpret_cast<char*>(hostOut.data()), hostOut.size());
    fout.close();

    std::cout << "Output written to " << outputFile << std::endl;

    aclDestroyDataBuffer(bufCipher); aclDestroyDataBuffer(bufKey); aclDestroyDataBuffer(bufOut);
    aclDestroyTensorDesc(descCipher); aclDestroyTensorDesc(descKey); aclDestroyTensorDesc(descOut);
    aclrtFree(devCipher); aclrtFree(devKey); aclrtFree(devOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    return 0;
}
