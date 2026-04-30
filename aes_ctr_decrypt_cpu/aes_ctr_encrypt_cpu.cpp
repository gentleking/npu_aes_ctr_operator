// aes_ctr_cpu.cpp
#include "aes_ctr_encrypt_cpu.h"
#include <cstring>

// 静态 S 盒初始化
static const uint8_t SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
    0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
    0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
    0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
    0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
    0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
    0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
    0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
    0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16
};

// GF(2^8) 乘法辅助函数
static uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00));
}

// 密钥扩展
void AesCtr::keyExpansion(const uint8_t* key, uint8_t* roundKeys) {
    uint8_t Rcon[11] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};

    // 复制原始密钥
    for (int i = 0; i < AES_KEY_BYTES; i++) {
        roundKeys[i] = key[i];
    }

    int bytesUsed = AES_KEY_BYTES;
    int rconIdx = 1;
    uint8_t temp[4];

    while (bytesUsed < AES_EXPANDED_KEY_BYTES) {
        // 取上一组的最后4字节
        for (int i = 0; i < 4; i++) {
            temp[i] = roundKeys[bytesUsed - 4 + i];
        }

        // 每16字节后，对 temp 进行轮变换
        if (bytesUsed % AES_KEY_BYTES == 0) {
            uint8_t t = temp[0];
            temp[0] = (uint8_t)(SBOX[temp[1]] ^ Rcon[rconIdx]);
            temp[1] = SBOX[temp[2]];
            temp[2] = SBOX[temp[3]];
            temp[3] = SBOX[t];
            rconIdx++;
        }

        // 生成新4字节
        for (int i = 0; i < 4; i++) {
            roundKeys[bytesUsed] = (uint8_t)(roundKeys[bytesUsed - AES_KEY_BYTES] ^ temp[i]);
            bytesUsed++;
        }
    }
}

// 轮密钥加
void AesCtr::addRoundKey(uint8_t* state, const uint8_t* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

// 字节代换
void AesCtr::subBytes(uint8_t* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = SBOX[state[i]];
    }
}

// 行移位
void AesCtr::shiftRows(uint8_t* state) {
    uint8_t tmp;
    // 第二行左移1位
    tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    // 第三行左移2位
    tmp = state[2];
    uint8_t tmp2 = state[6];
    state[2] = state[10];
    state[6] = state[14];
    state[10] = tmp;
    state[14] = tmp2;

    // 第四行左移3位
    tmp = state[3];
    state[3] = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = tmp;
}

// 列混合
void AesCtr::mixColumns(uint8_t* state) {
    for (int col = 0; col < 4; col++) {
        int idx = col * 4;
        uint8_t a0 = state[idx + 0];
        uint8_t a1 = state[idx + 1];
        uint8_t a2 = state[idx + 2];
        uint8_t a3 = state[idx + 3];

        uint8_t t = a0 ^ a1 ^ a2 ^ a3;
        uint8_t u = a0;
        state[idx + 0] ^= t ^ xtime((uint8_t)(a0 ^ a1));
        state[idx + 1] ^= t ^ xtime((uint8_t)(a1 ^ a2));
        state[idx + 2] ^= t ^ xtime((uint8_t)(a2 ^ a3));
        state[idx + 3] ^= t ^ xtime((uint8_t)(a3 ^ u));
    }
}

// 单个块加密（前9轮包含 MixColumns，最后一轮不包含）
void AesCtr::encryptBlock(const uint8_t in[16], const uint8_t roundKeys[176], uint8_t out[16]) {
    uint8_t state[16];
    memcpy(state, in, 16);

    // 初始轮密钥加
    addRoundKey(state, roundKeys);

    // 前9轮
    for (int r = 1; r < AES_ROUND_NUM; r++) {
        subBytes(state);
        shiftRows(state);
        mixColumns(state);
        addRoundKey(state, roundKeys + 16 * r);
    }

    // 最后一轮（无 MixColumns）
    subBytes(state);
    shiftRows(state);
    addRoundKey(state, roundKeys + 16 * AES_ROUND_NUM);

    memcpy(out, state, 16);
}

// 大端加法（在 128 位计数器上增加 val）
void AesCtr::bigEndAdd(uint8_t* ctr, uint32_t val) {
    int carry = 0;
    for (int i = 15; i >= 0; --i) {
        uint32_t sum = (uint32_t)ctr[i] + (val & 0xFF) + carry;
        ctr[i] = (uint8_t)(sum & 0xFF);
        carry = (sum >> 8) & 0xFF;
        val >>= 8;
        if (val == 0 && carry == 0) break;
    }
}

// 生成一个密钥流块（加密当前计数器）
void AesCtr::generateKeystream(uint8_t* keystream) {
    encryptBlock(ctr_, roundKeys_, keystream);
    // 计数器递增（大端加法）
    bigEndAdd(ctr_, 1);
}

// 构造函数
AesCtr::AesCtr(const uint8_t key[16], const uint8_t iv[16]) {
    // 初始化 S 盒
    for (int i = 0; i < 256; i++) {
        sbox_[i] = SBOX[i];
    }
    // 密钥扩展
    keyExpansion(key, roundKeys_);
    // 保存初始 IV 并初始化当前计数器
    memcpy(iv_, iv, 16);
    memcpy(ctr_, iv, 16);
}

// 重置计数器
void AesCtr::reset() {
    memcpy(ctr_, iv_, 16);
}

// 处理数据
void AesCtr::process(const uint8_t* in, uint8_t* out, int len) {
    uint8_t keystream[AES_BLOCK_BYTES];
    int offset = 0;
    while (offset < len) {
        generateKeystream(keystream);
        int blockSize = (len - offset < AES_BLOCK_BYTES) ? (len - offset) : AES_BLOCK_BYTES;
        for (int i = 0; i < blockSize; i++) {
            out[offset + i] = in[offset + i] ^ keystream[i];
        }
        offset += blockSize;
    }
}