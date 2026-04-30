// aes_ctr_cpu.h
#ifndef AES_CTR_CPU_H
#define AES_CTR_CPU_H

#include <cstdint>

// AES 参数定义
static const int AES_KEY_BYTES = 16;                // AES-128 密钥字节数
static const int AES_BLOCK_BYTES = 16;              // AES 分组大小
static const int AES_ROUND_NUM = 10;                // AES 轮数
static const int AES_EXPANDED_KEY_BYTES = 16 * (AES_ROUND_NUM + 1);  // 扩展密钥字节数

/**
 * @brief AES-128 CTR 模式加密/解密（CTR 模式下加解密相同）
 */
class AesCtr {
public:
    /**
     * @brief 构造函数
     * @param key 16 字节的 AES-128 密钥
     * @param iv  16 字节的初始计数器（大端表示）
     */
    AesCtr(const uint8_t key[16], const uint8_t iv[16]);

    /**
     * @brief 对任意长度数据进行 CTR 模式加密/解密
     * @param in   输入数据缓冲区
     * @param out  输出数据缓冲区（可以与 in 相同）
     * @param len  数据长度（字节）
     */
    void process(const uint8_t* in, uint8_t* out, int len);

    /**
     * @brief 重置计数器为初始 IV
     */
    void reset();

private:
    // AES 核心函数
    static void keyExpansion(const uint8_t* key, uint8_t* roundKeys);
    static void addRoundKey(uint8_t* state, const uint8_t* roundKey);
    static void subBytes(uint8_t* state);
    static void shiftRows(uint8_t* state);
    static void mixColumns(uint8_t* state);
    static void encryptBlock(const uint8_t in[16], const uint8_t roundKeys[176], uint8_t out[16]);

    // CTR 模式辅助
    static void bigEndAdd(uint8_t* ctr, uint32_t val);
    void generateKeystream(uint8_t* keystream);

    uint8_t roundKeys_[AES_EXPANDED_KEY_BYTES];   // 扩展密钥
    uint8_t iv_[AES_BLOCK_BYTES];                 // 初始 IV
    uint8_t ctr_[AES_BLOCK_BYTES];                // 当前计数器
    uint8_t sbox_[256];                           // S 盒（固定）
};

#endif // AES_CTR_CPU_H