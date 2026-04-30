/**
 * @file aes_ctr_decrypt.cpp
 * @brief Ascend C 自定义AES-CTR解密算子
 *
 * 说明：
 * - 本代码实现了AES-CTR模式下的解密算子，其中AES使用128位密钥进行加密（10轮变换）。
 * - 核心模块包括密钥扩展(KeyExpansion)、单块AES加密(AesEncryptBlock)和CTR模式加密(AesCtrEncrypt)。
 * - 针对NPU模式和CPU Debug模式分别实现了不同版本的入口函数。
 *
 * 手册提供的LocalTensor<T>主要支持：
 *   - GetValue(offset)、SetValue(offset, val)
 *   - operator[](offset) / operator()(offset)
 *   - GetSize() / SetSize()
 *   - DataCopy(LocalTensor<T>, GlobalTensor<T>, count) / DataCopy(GlobalTensor<T>, LocalTensor<T>, count)
 */

#include "kernel_operator.h"
#include <cstring>

 // ------------------ 常量/宏定义 ------------------
constexpr int32_t TOTAL_LENGTH = 8 * 1024;  // 总half数量（128bits）
constexpr int32_t USE_CORE_NUM = 8;         // 使用的核心数
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;             // tile数量 8
constexpr int32_t BUFFER_NUM = 2;           // 双缓冲数量
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // pipeline切分长度128
 
 // AES 参数定义
 static const int AES_KEY_BYTES = 16;                                // AES-128密钥字节数
 static const int AES_BLOCK_BYTES = 16;                              // AES分组大小
 static const int AES_ROUND_NUM = 10;                                // AES轮数
 static const int AES_EXPANDED_KEY_BYTES = 16 * (AES_ROUND_NUM + 1);   // 扩展密钥字节数
 
 /**
  * @brief GF(2^8)中左移操作，若最高位为1则异或0x1B
  */
 __aicore__ inline static uint8_t xtime(uint8_t x)
 {
     return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00));
 }
 
 /**
  * @brief AES-128的密钥扩展，将原始16字节密钥扩展为176字节
  * @param key 原始密钥，长度为16字节
  * @param roundKeys 输出扩展密钥缓冲区，长度为176字节
  * @param sbox 传入的AES S-box数组
  */
static __aicore__ void KeyExpansion(const uint8_t *key, uint8_t *roundKeys, const uint8_t *sbox)
{
    // Rcon常量用于轮密钥扩展
    uint8_t Rcon[11] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
    // 复制原始密钥到扩展密钥缓冲区
    for (int i = 0; i < AES_KEY_BYTES; i++) {
        roundKeys[i] = key[i];
    }

    int bytesUsed = AES_KEY_BYTES; // 当前已生成的字节数
    int rconIdx = 1;               // Rcon索引
    volatile uint8_t temp[4];

    // 迭代生成剩余的扩展密钥
    while (bytesUsed < AES_EXPANDED_KEY_BYTES) {
        // 取上一组4字节
        for (int i = 0; i < 4; i++) {
            temp[i] = roundKeys[bytesUsed - 4 + i];
        }
        // 每16字节后，对temp进行轮变换
        if ((bytesUsed % AES_KEY_BYTES) == 0) {
            // 轮变换：RotWord, SubWord, 并与Rcon异或
            uint8_t t = temp[0];
            temp[0] = (uint8_t)(sbox[temp[1]] ^ Rcon[rconIdx]);
            temp[1] = sbox[temp[2]];
            temp[2] = sbox[temp[3]];
            temp[3] = sbox[t];
            rconIdx++;
        }
        // 生成新4字节子密钥
        for (int i = 0; i < 4; i++) {
            roundKeys[bytesUsed] = (uint8_t)(roundKeys[bytesUsed - AES_KEY_BYTES] ^ temp[i]);
            bytesUsed++;
        }
    }
}
 
/**
 * @brief 将状态数组与子密钥进行异或
 * @param state 状态数组（16字节）
 * @param roundKey 当前轮的子密钥（16字节）
 */
static __aicore__ inline void AddRoundKey(uint8_t *state, const uint8_t *roundKey)
{
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];//
    }
}

/**
 * @brief 前9轮的AES变换：SubBytes、ShiftRows和MixColumns
 * @param state 状态数组（16字节）
 * @param blockId 当前块ID（调试或扩展时使用，可用于区分不同块的处理）
 * @param sbox 传入的AES S-box数组
 */
static __aicore__ void SubShiftMix(uint8_t *state, uint32_t blockId, const uint8_t *sbox)
{
    // 1) SubBytes：用传入的 sbox 替换每个字节
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }

    // 2) ShiftRows：行移位变换
    {
        uint8_t tmp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = tmp;

        tmp = state[2];
        uint8_t tmp2 = state[6];
        state[2] = state[10];
        state[6] = state[14];
        state[10] = tmp;
        state[14] = tmp2;

        tmp = state[3];
        state[3] = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = tmp;
    }

    // 3) MixColumns：列混合变换
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

/**
 * @brief 最后一轮的AES变换：SubBytes和ShiftRows（不执行MixColumns）
 * @param state 状态数组（16字节）
 * @param sbox 传入的AES S-box数组
 */
static __aicore__ void SubShift(uint8_t *state, const uint8_t *sbox)
{
    // 1) SubBytes
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }

    // 2) ShiftRows
    {
        uint8_t tmp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = tmp;

        tmp = state[2];
        uint8_t tmp2 = state[6];
        state[2] = state[10];
        state[6] = state[14];
        state[10] = tmp;
        state[14] = tmp2;

        tmp = state[3];
        state[3] = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = tmp;
    }
}
 
/**
 * @brief 单块AES-128加密函数
 * @param in  输入明文（16字节）
 * @param roundKeys 扩展密钥（176字节）
 * @param out 输出密文（16字节）
 * @param blockId 当前块ID，用于调试或特殊处理（可忽略）
 * @param sbox 传入的AES S-box数组
 */
static __aicore__ void AesEncryptBlock(const uint8_t in[16], const uint8_t roundKeys[AES_EXPANDED_KEY_BYTES],
                                    uint8_t out[16], uint32_t blockId, const uint8_t *sbox)
{
    uint8_t state[16];
    // 初始化状态
    for (int i = 0; i < 16; i++) {
        state[i] = in[i];
    }
    // 初始轮：状态与初始子密钥异或
    AddRoundKey(state, roundKeys);

    // 前9轮
    for (int r = 1; r < AES_ROUND_NUM; r++) {
        SubShiftMix(state, blockId, sbox);
        AddRoundKey(state, roundKeys + 16 * r);
    }
    // 最后一轮（无MixColumns）
    SubShift(state, sbox);
    AddRoundKey(state, roundKeys + 16 * AES_ROUND_NUM);

    // 输出结果
    for (int i = 0; i < 16; i++) {
        out[i] = state[i];
    }
}

/**
 * @brief 大端加法：将val加到表示为大端格式的128位数ctr中
 * @param ctr 计数器数组（16字节，大端格式）
 * @param val 要加的数值
 */
__aicore__ inline void BigEndAdd(uint8_t *ctr, uint32_t val)
{
    int carry = 0;
    for (int i = 15; i >= 0; --i) {
        // 计算当前字节的和：当前ctr字节 + 当前val字节（val & 0xFF） + 进位
        uint32_t sum = (uint32_t)ctr[i] + (val & 0xFF) + carry;
        // 将结果的低8位保存回ctr[i]
        ctr[i] = (uint8_t)(sum & 0xFF);
        // 提取进位：高于8位的部分
        carry = (sum >> 8) & 0xFF;
        // 右移val，使下一次循环处理val的下一个字节
        val >>= 8;
        // 如果没有剩余的val和进位为0，则可提前退出循环
        if (val == 0 && carry == 0) {
            break;
        }
    }
}

 
 /**
  * @brief AES-CTR模式加密
  *
  * 对输入明文以16字节为一组进行加密，每组加密时：
  * - 使用当前ctrBlock生成密钥流
  * - 对应明文与密钥流异或得到密文
  * - 计数器ctrBlock自增（大端加法）
  *
  * @param plaintext 输入明文
  * @param ciphertext 输出密文
  * @param lengthBytes 明文字节数
  * @param roundKeys 扩展密钥（176字节）
  * @param iv 初始向量（16字节，处理后更新为最新的计数器状态）
  * @param blockId 当前块ID
  * @param sbox 传入的AES S-box数组
  */
static __aicore__ void AesCtrEncrypt(const uint8_t *plaintext, uint8_t *ciphertext, int32_t lengthBytes,
                                    const uint8_t *roundKeys, uint8_t iv[16], int32_t blockId, const uint8_t *sbox)
{
    uint8_t ctrBlock[16];
    // 初始化ctrBlock为当前iv
    for (int i = 0; i < 16; i++) {
        ctrBlock[i] = iv[i];
    }

    uint8_t keystream[16];
    int offset = 0;
    while (offset < lengthBytes) {
        // 生成当前计数器下的密钥流
        AesEncryptBlock(ctrBlock, roundKeys, keystream, blockId, sbox);

        int blockSize = (lengthBytes - offset < 16) ? (lengthBytes - offset) : 16;
        // 明文与密钥流异或得到密文
        for (int i = 0; i < blockSize; i++) {
            ciphertext[offset + i] = plaintext[offset + i] ^ keystream[i];
        }
        offset += blockSize;

        // 计数器自增（大端顺序）
        for (int i = 15; i >= 0; i--) {
            ctrBlock[i]++;
            if (ctrBlock[i] != 0) {
                break;
            }
        }
    }
    // 更新iv为最终的计数器状态
    for (int i = 0; i < 16; i++) {
        iv[i] = ctrBlock[i];
    }
}
 
#ifndef ASCENDC_CPU_DEBUG

// ------------------ NPU模式下：TQue + 多缓冲 Pipeline ------------------
/**
 * @brief KernelAesCtr类：在Ascend C环境（NPU模式）下实现AES-CTR的多缓冲流水线加密
 */
class KernelAesCtrDecrypt {
    public:
        /**
         * @brief 构造函数：初始化AES的S-box
         */
        __aicore__ inline KernelAesCtrDecrypt()
        {
            // 初始化 sbox 数组（固定常量）
            uint8_t sbox_init[256] = {
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
            // 将常量数组拷贝到私有成员 sbox 中
            for (int i = 0; i < 256; i++) {
                sbox[i] = sbox_init[i];
            }
        }
    
        /**
         * @brief 初始化：设置GM全局缓冲区，初始化双缓冲队列，并从GM复制密钥和IV，执行密钥扩展
         * @param plaintext GM中存储明文的地址
         * @param key       GM中存储AES-128密钥的地址
         * @param iv        GM中存储初始向量(IV)的地址
         * @param ciphertext GM中存储加密结果的地址
         *
         * - 通过 GlobalTensor 设置 plaintextGm、ciphertextGm 的起始地址和长度。
         * - 调用 TPipe::InitBuffer() 初始化多缓冲队列，以实现流水线并行。
         * - 从GM复制key、iv到本地，然后调用KeyExpansion进行密钥扩展。
         * - 根据blockIdx对IV做偏移处理，以便多核并行时能使用不同的计数器起点。
         */
        __aicore__ inline void Init(GM_ADDR plaintext, GM_ADDR key, GM_ADDR ciphertext, uint32_t totalLength, uint32_t tileNum)
        {
            // this->USE_CORE_NUM = AscendC::GetBlockNum();
            // this->TOTAL_LENGTH = totalLength;
            // this->TILE_NUM = tileNum;
            // this->BLOCK_LENGTH = this->TOTAL_LENGTH / this->USE_CORE_NUM;
            // this->TILE_LENGTH = this->BLOCK_LENGTH / this->TILE_NUM / BUFFER_NUM;

            int blockIdx = AscendC::GetBlockIdx();

            // 配置全局内存缓冲区
            plaintextGm.SetGlobalBuffer((__gm__ half *)plaintext + BLOCK_LENGTH * blockIdx, BLOCK_LENGTH);
            ciphertextGm.SetGlobalBuffer((__gm__ half *)ciphertext + BLOCK_LENGTH * blockIdx, BLOCK_LENGTH);
    
            // 初始化双缓冲队列
            pipe.InitBuffer(inQueuePlain, BUFFER_NUM, TILE_LENGTH * sizeof(half));
            pipe.InitBuffer(outQueueCipher, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    
            // 从GM复制密钥到localKey，并进行扩展
            uint8_t localKey[AES_KEY_BYTES];
            volatile __gm__ uint8_t *gm_key = key;
            for (int i = 0; i < AES_KEY_BYTES; i++) {
                localKey[i] = gm_key[i];
            }
    
            // 从GM复制IV到本地IV
            // uint8_t init_iv[16] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
            //                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}; // 用于保存初始IV状态
     
            // volatile __gm__ uint8_t *gm_iv = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
            //                                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}; // 用于保存初始IV状态;
            // 初始化IV为全0
            for (int i = 0; i < AES_BLOCK_BYTES; i++) {
                // IV[i] = gm_iv[i];
                IV[i] = 0;
            }
    
            // 执行AES-128密钥扩展
            KeyExpansion(localKey, expandedKey, sbox);
    
            // 根据blockIdx调整IV，使多核处理使用不同的初始计数器
            // (这里的offsetBlocks计算逻辑可根据实际分块大小进行修改)
            uint32_t offsetBlocks = (BLOCK_LENGTH * 2) / AES_BLOCK_BYTES;
            uint8_t tmpIV[16];
            for (int i = 0; i < 16; i++) {
                tmpIV[i] = IV[i];
            }
            BigEndAdd(tmpIV, offsetBlocks * blockIdx);
            for (int i = 0; i < 16; i++) {
                IV[i] = tmpIV[i];
            }
        }
    
        /**
         * @brief 处理入口：依次执行数据读入、加密计算、数据写出
         *
         * 其中包含TILE_NUM * BUFFER_NUM次循环，每次循环都完成一次CopyIn、Compute、CopyOut的排布。
         * 在硬件层面，CopyIn、Compute、CopyOut三个阶段可以流水线并行执行，从而提升吞吐量。
         */
        __aicore__ inline void Process()
        {
            int loopCount = TILE_NUM * BUFFER_NUM;
            for (int i = 0; i < loopCount; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    
    private:
        /**
         * @brief 数据读入：从GM复制数据到本地Tensor
         *
         * 流水线Stage 1：本函数对应“CopyIn”阶段，将全局内存（GM）中的数据搬运到Local Memory。
         * - 使用 TQue 的 AllocTensor<half>() 分配一个可存储 TILE_LENGTH 大小的本地Tensor。
         * - 使用 DataCopy() 将 GM 中的明文数据复制到本地Tensor。
         * - 将装载了数据的本地Tensor 放入（EnQue）队列 inQueuePlain，供下一个Stage（Compute）使用。
         *
         * 参考官方开发手册中 “Vector+队列模式” 中对 TQue 的使用和 DataCopy 的描述。
         */
        __aicore__ inline void CopyIn(int idx)
        {
            AscendC::LocalTensor<half> plainLocal = inQueuePlain.AllocTensor<half>();
            AscendC::DataCopy(plainLocal, plaintextGm[idx * TILE_LENGTH], TILE_LENGTH);
            inQueuePlain.EnQue(plainLocal);
        }
    
        /**
         * @brief 核心加密计算：从本地Tensor读取数据，执行AES-CTR加密，再写回加密结果到本地Tensor
         *
         * 流水线Stage 2：本函数对应“Compute”阶段，读取队列 inQueuePlain 中的本地Tensor，
         * 进行AES-CTR加密，然后将结果写入另一个本地Tensor，再放入 outQueueCipher 队列，等待下一阶段输出。
         *
         * - DeQue() 从 inQueuePlain 中获取上一阶段传入的明文Tensor。
         * - 分配新的本地Tensor（cipherLocal）以存放加密结果。
         * - 进行必要的数据格式转换（half -> uint8_t 数组），再调用 AesCtrEncrypt。
         * - 将加密结果转换回 half 并写入 cipherLocal。
         * - EnQue() 将加密后的数据放入 outQueueCipher。
         * - FreeTensor() 释放用完的明文Tensor 以复用本地内存。
         *
         * 参考官方开发手册 “流水线编程范式”的多缓冲（Double Buffer）与队列操作示例。
         */
        __aicore__ inline void Compute(int idx)
        {
            int32_t blockIdx = AscendC::GetBlockIdx();
            AscendC::LocalTensor<half> plainLocal = inQueuePlain.DeQue<half>();
            AscendC::LocalTensor<half> cipherLocal = outQueueCipher.AllocTensor<half>();
    
            // 复制当前IV到临时缓冲区
            uint8_t TmpIv[16];
            for (int i = 0; i < 16; i++) {
                TmpIv[i] = IV[i];
            }
            constexpr int tileBytes = TILE_LENGTH * 2;
            uint8_t plainBuf[tileBytes];
            // 将half格式数据转换为字节数组
            for (int i = 0; i < TILE_LENGTH; i++) {
                half hVal = plainLocal.GetValue(i);
                uint16_t halfBits = *reinterpret_cast<uint16_t *>(&hVal);
                plainBuf[2 * i + 0] = static_cast<uint8_t>(halfBits & 0xFF);
                plainBuf[2 * i + 1] = static_cast<uint8_t>((halfBits >> 8) & 0xFF);
            }
    
            // 执行AES-CTR加密
            uint8_t cipherBuf[tileBytes];
            AesCtrEncrypt(plainBuf, cipherBuf, tileBytes, expandedKey, TmpIv, blockIdx, sbox);
    
            // 更新本地IV
            for (int i = 0; i < 16; i++) {
                IV[i] = TmpIv[i];
            }
    
            // 将加密后的字节数组转换回half格式，并写入本地Tensor
            for (int i = 0; i < TILE_LENGTH; i++) {
                uint16_t halfBits = static_cast<uint16_t>(cipherBuf[2 * i]) |
                                    (static_cast<uint16_t>(cipherBuf[2 * i + 1]) << 8);
                half val = *reinterpret_cast<half *>(&halfBits);
                cipherLocal.SetValue(i, val);
            }
    
            outQueueCipher.EnQue<half>(cipherLocal);
            inQueuePlain.FreeTensor(plainLocal);
        }
    
        /**
         * @brief 数据写出：将加密结果从本地Tensor复制回GM
         *
         * 流水线Stage 3：本函数对应“CopyOut”阶段，从 outQueueCipher 中取出已经完成加密的本地Tensor，
         * 并使用 DataCopy() 写回至 GM 中的目标地址，最后释放本地Tensor。
         *
         * - DeQue() 从 outQueueCipher 获取已加密的本地Tensor。
         * - DataCopy() 将加密结果写回到 GM（ciphertextGm）。
         * - FreeTensor() 释放本地Tensor 以复用空间。
         *
         * 至此，一个完整的 Pipeline（CopyIn -> Compute -> CopyOut）操作完成。
         */
        __aicore__ inline void CopyOut(int idx)
        {
            AscendC::LocalTensor<half> cipherLocal = outQueueCipher.DeQue<half>();
            AscendC::DataCopy(ciphertextGm[idx * TILE_LENGTH], cipherLocal, TILE_LENGTH);
            outQueueCipher.FreeTensor(cipherLocal);
        }
    
    private:
        AscendC::TPipe pipe;  ///< 负责管理Pipeline的调度与多缓冲机制
    
        /// 输入队列：存放明文的本地Tensor（VECIN 位置）
        AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePlain;
        /// 输出队列：存放加密后密文的本地Tensor（VECOUT 位置）
        AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueCipher;
    
        /// GM中存储明文的区域
        AscendC::GlobalTensor<half> plaintextGm;
        /// GM中存储密文的区域
        AscendC::GlobalTensor<half> ciphertextGm;
    
        /// 存储GM传入的iv和key及扩展后的密钥
        uint8_t IV[AES_BLOCK_BYTES];
        uint8_t expandedKey[AES_EXPANDED_KEY_BYTES];
    
        /// AES S-box数组，用于SubBytes等操作
        uint8_t sbox[256];

        // int32_t TOTAL_LENGTH;  // 总数量
        // int32_t USE_CORE_NUM;         // 使用的核心数
        // int32_t BLOCK_LENGTH;
        // int32_t TILE_NUM;             // tile数量 8
        // int32_t TILE_LENGTH; // pipeline切分长度128
        // uint32_t totalLength;
        // uint32_t tileNum;
        // uint32_t blockLength;
        // uint32_t tileLength;
    };
    

 
extern "C" __global__ __aicore__ void aes_ctr_decrypt(GM_ADDR cyphertext, GM_ADDR key, GM_ADDR plaintext, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    KernelAesCtrDecrypt op;
    op.Init(cyphertext, key, plaintext, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

void aes_ctr_decrypt_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *ciphertext, uint8_t *key, uint8_t *plaintext, 
                         uint8_t *workspace, uint8_t *tiling) {
    aes_ctr_decrypt<<<blockDim, l2ctrl, stream>>>(ciphertext, key, plaintext);
}

#endif