#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include "acl/acl.h"

static void check_acl(const char* msg, aclError ret) {
    if (ret != ACL_SUCCESS) {
        std::cerr << msg << " failed: " << ret << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    int dataSize = 1073741824;
    int numWarmup = 1;
    int numRuns = 3;
    int deviceId = 0;
    std::string modelDir = "op_models";
    std::string cipherFile = "input/ciphertext.bin";
    std::string keyFile = "input/key.bin";
    std::string resultFile = "output/result.bin";

    if (argc > 1) dataSize = std::atoi(argv[1]);
    if (argc > 2) numWarmup = std::atoi(argv[2]);
    if (argc > 3) numRuns = std::atoi(argv[3]);
    if (argc > 4) modelDir = argv[4];
    if (argc > 5) deviceId = std::atoi(argv[5]);

    check_acl("aclInit", aclInit("scripts/acl.json"));
    check_acl("aclopSetModelDir", aclopSetModelDir(modelDir.c_str()));
    check_acl("aclrtSetDevice", aclrtSetDevice(deviceId));

    aclrtRunMode runMode;
    check_acl("aclrtGetRunMode", aclrtGetRunMode(&runMode));
    bool isDevice = (runMode == ACL_DEVICE);

    // Allocate device memory (single buffer for both input and output => in-place)
    void *devBuf = nullptr, *devKey = nullptr;

    aclError r = aclrtMalloc(&devBuf, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r != ACL_SUCCESS || !devBuf)
        r = aclrtMalloc(&devBuf, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (r != ACL_SUCCESS || !devBuf) {
        std::cerr << "Cannot allocate " << dataSize << " bytes on device" << std::endl;
        return 1;
    }

    r = aclrtMalloc(&devKey, 16, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r != ACL_SUCCESS || !devKey)
        r = aclrtMalloc(&devKey, 16, ACL_MEM_MALLOC_NORMAL_ONLY);
    check_acl("aclrtMalloc devKey", r);

    // Use mmap for large host buffers instead of heap allocation
    size_t pageSize = sysconf(_SC_PAGESIZE);
    size_t mapSize = ((dataSize + pageSize - 1) / pageSize) * pageSize;

    uint8_t* hostBuf = static_cast<uint8_t*>(
        mmap(nullptr, mapSize, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    uint8_t* hostOut = static_cast<uint8_t*>(
        mmap(nullptr, mapSize, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    uint8_t* hostKey = new uint8_t[16];

    if (hostBuf == MAP_FAILED || hostOut == MAP_FAILED) {
        std::cerr << "mmap failed for " << dataSize << " bytes" << std::endl;
        return 1;
    }

    std::ifstream fCipher(cipherFile, std::ios::binary);
    if (!fCipher) { std::cerr << "Cannot open " << cipherFile << std::endl; return 1; }
    fCipher.read(reinterpret_cast<char*>(hostBuf), dataSize);

    std::ifstream fKey(keyFile, std::ios::binary);
    fKey.read(reinterpret_cast<char*>(hostKey), 16);

    aclrtMemcpyKind h2d = isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE;
    aclrtMemcpyKind d2h = isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;

    check_acl("aclrtMemcpy data", aclrtMemcpy(devBuf, dataSize, hostBuf, dataSize, h2d));
    check_acl("aclrtMemcpy key", aclrtMemcpy(devKey, 16, hostKey, 16, h2d));

    std::vector<int64_t> inShape = {dataSize};
    std::vector<int64_t> keyShape = {16};
    std::vector<int64_t> outShape = {dataSize};

    aclTensorDesc *descIn = aclCreateTensorDesc(ACL_UINT8, 1, inShape.data(), ACL_FORMAT_ND);
    aclTensorDesc *descKey = aclCreateTensorDesc(ACL_UINT8, 1, keyShape.data(), ACL_FORMAT_ND);
    aclTensorDesc *descOut = aclCreateTensorDesc(ACL_UINT8, 1, outShape.data(), ACL_FORMAT_ND);

    aclDataBuffer *bufIn = aclCreateDataBuffer(devBuf, dataSize);
    aclDataBuffer *bufK = aclCreateDataBuffer(devKey, 16);
    aclDataBuffer *bufOut = aclCreateDataBuffer(devBuf, dataSize);

    aclrtStream stream;
    check_acl("aclrtCreateStream", aclrtCreateStream(&stream));

    aclTensorDesc *inputsDesc[2] = {descIn, descKey};
    aclDataBuffer *inputsBuf[2] = {bufIn, bufK};
    aclDataBuffer *outputsBuf[1] = {bufOut};
    aclTensorDesc *outputsDesc[1] = {descOut};

    for (int i = 0; i < numWarmup; i++) {
        check_acl("aclopExecuteV2 warmup",
            aclopExecuteV2("AesCtrDecryptAicpu", 2, inputsDesc, inputsBuf,
                           1, outputsDesc, outputsBuf, nullptr, stream));
        check_acl("aclrtSynchronizeStream warmup",
            aclrtSynchronizeStream(stream));
    }

    std::vector<double> times;
    times.reserve(numRuns);
    for (int i = 0; i < numRuns; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        check_acl("aclopExecuteV2",
            aclopExecuteV2("AesCtrDecryptAicpu", 2, inputsDesc, inputsBuf,
                           1, outputsDesc, outputsBuf, nullptr, stream));
        check_acl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    check_acl("aclrtMemcpy result",
        aclrtMemcpy(hostOut, dataSize, devBuf, dataSize, d2h));

    std::ofstream fout(resultFile, std::ios::binary);
    fout.write(reinterpret_cast<char*>(hostOut), dataSize);
    fout.close();

    double sum = 0, min = times[0], max = times[0];
    for (double t : times) { sum += t; if (t < min) min = t; if (t > max) max = t; }
    double avg = sum / numRuns;

    std::cout << "DATA_SIZE=" << dataSize
              << " RUNS=" << numRuns
              << " AVG_MS=" << avg
              << " MIN_MS=" << min
              << " MAX_MS=" << max
              << " THROUGHPUT_MBPS=" << (dataSize / 1024.0 / 1024.0) / (avg / 1000.0)
              << std::endl;

    aclDestroyDataBuffer(bufIn); aclDestroyDataBuffer(bufK); aclDestroyDataBuffer(bufOut);
    aclDestroyTensorDesc(descIn); aclDestroyTensorDesc(descKey); aclDestroyTensorDesc(descOut);
    aclrtFree(devBuf); aclrtFree(devKey);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId); aclFinalize();
    munmap(hostBuf, mapSize); munmap(hostOut, mapSize); delete[] hostKey;

    return 0;
}
