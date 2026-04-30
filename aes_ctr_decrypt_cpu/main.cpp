// test_aes_ctr.cpp
#include <iostream>
#include <iomanip>
#include <cstring>
#include "aes_ctr_encrypt_cpu.h"

// 辅助函数：打印字节数组
void printHex(const uint8_t* data, int len, const std::string& title) {
    std::cout << title << ": ";
    for (int i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

int main() {
    // 测试向量（NIST SP 800-38A 附录 B）
    uint8_t key[16] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
    };
    uint8_t iv[16] = {
        0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
        0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff
    };
    // 明文（第一块）
    uint8_t plaintext[] = {
        0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
        0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
        0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
        0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a
    };
    // 期望的密文（第一块）
    uint8_t expected_cipher[] = {
        0x87, 0x4d, 0x61, 0x91, 0xb6, 0x20, 0xe3, 0x26,
        0x1b, 0xef, 0x68, 0x64, 0x99, 0x0d, 0xb6, 0xce,
        0x87, 0x4d, 0x61, 0x91, 0xb6, 0x20, 0xe3, 0x26,
        0x1b, 0xef, 0x68, 0x64, 0x99, 0x0d, 0xb6, 0xce
    };

    // 加密
    AesCtr aes(key, iv);
    uint8_t cipher[32];
    aes.process(plaintext, cipher, 32);
    printHex(expected_cipher, 32, "期望结果");
    printHex(cipher, 32, "加密结果");

    // 验证
    if (memcmp(cipher, expected_cipher, 32) == 0) {
        std::cout << "测试通过：加密结果与预期一致。" << std::endl;
    } else {
        std::cout << "测试失败：加密结果与预期不符。" << std::endl;
    }

    // 解密（CTR 模式加解密相同，所以再次调用 process 应得到原文）
    uint8_t decrypted[32];
    aes.reset();                     // 重置计数器
    aes.process(cipher, decrypted, 32);
    printHex(plaintext, 32, "明文结果");
    printHex(decrypted, 32, "解密结果");

    if (memcmp(decrypted, plaintext, 32) == 0) {
        std::cout << "解密正确，恢复明文" << std::endl;
    } else {
        std::cout << "解密失败。" << std::endl;
    }

    // 测试多块数据（完整 NIST 测试，可选）
    // 此处仅演示第一块，更多块可自行扩展

    return 0;
}