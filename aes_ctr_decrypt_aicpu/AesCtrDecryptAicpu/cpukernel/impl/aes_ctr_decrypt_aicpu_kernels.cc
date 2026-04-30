/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: AI CPU 自定义算子 AesCtrDecryptAicpu 的核函数实现
 *
 * 本文件实现了 AES-128 CTR 模式的完整解密算法，包括：
 *   1. AES 核心算法: SubBytes, ShiftRows, MixColumns, AddRoundKey, KeyExpansion
 *   2. CTR 模式: 计数器管理、密钥流生成、密文异或解密
 *
 * 算法流程:
 *   Compute() 入口 ->
 *   初始化 S 盒、密钥扩展 -> 设置 IV 和计数器 ->
 *   循环: generateKeystream -> 密文 XOR 密钥流 -> offset 前进 ->
 *   每 16 字节一个分组，计数器递增
 */
#include "aes_ctr_decrypt_aicpu_kernels.h"

// ====================================================================
// 匿名命名空间: 内部链接，避免多文件编译时符号冲突
// 这些常量和变量仅在本编译单元可见
// ====================================================================
namespace {

// 算子类型名称字符串，用于 REGISTER_CPU_KERNEL 宏
// 必须与算子原型定义 (json) 中的 op 名称一致
const char *AES_CTR_DECRYPT_AICPU = "AesCtrDecryptAicpu";

// Tensor 索引常量
const uint32_t kFirstInputIndex = 0;   // 第一个输入: 密文 (ciphertext)
const uint32_t kSecondInputIndex = 1;  // 第二个输入: 密钥 (key)
const uint32_t kFirstOutputIndex = 0;  // 输出: 明文 (plaintext)

// 返回值状态码
const uint32_t SUCCESS = 0;         // 算子执行成功
const uint32_t PARAM_INVAILD = 1;   // 参数无效
const uint32_t ERROR = 2;           // 其他错误

// 初始向量 IV: 固定为全零
// 在生产环境中，IV 通常作为算子输入传入，本算子为简化固定为全零
const uint8_t iv[16] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

/*
 * AES S 盒: 16x16 查找表，用于 SubBytes（字节代换）步骤
 * S 盒是 AES 算法中唯一的非线性变换，提供混淆效果。
 * 数值 = GF(2^8) 上的乘法逆元经过仿射变换后的结果。
 *
 * 索引方式: high_nibble = state[i] >> 4, low_nibble = state[i] & 0x0F
 */
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

}  // 匿名命名空间结束

/*
 * xtime: GF(2^8) 域上的乘以 x（即乘以 2）运算
 *
 * 在 AES MixColumns 中使用。
 * 规则: 左移 1 位; 如果原值最高位为 1，则需异或 0x1B 以模
 *       AES 不可约多项式 x^8 + x^4 + x^3 + x + 1
 *
 * 参数 x: 待运算的 8 位值
 * 返回值: x * 0x02 (在 GF(2^8) 中)
 */
static uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00));
}

namespace aicpu {

// ====================================================================
// Compute: AICPU 框架入口
//
// 此函数在 AICPU 调度器收到 Host 的算子执行请求后被调用。
// 框架会将算子描述中的输入输出 tensor、属性等信息包装为
// CpuKernelContext 传入。
//
// CTR 解密流程:
//   1. 初始化 S 盒、扩展密钥、IV/计数器
//   2. 循环处理每个 16 字节分组:
//      a. 加密当前计数器 -> 密钥流
//      b. 计数器 +1
//      c. 密文 XOR 密钥流 -> 明文
// ====================================================================
uint32_t AesCtrDecryptAicpuCpuKernel::Compute(CpuKernelContext &ctx)
{
    // --- 1. 获取输入输出 tensor ---
    // ctx.Input(index) / ctx.Output(index) 返回 Tensor*
    // 如果 tensor 不存在或数据为空，可能返回 nullptr
    Tensor *input0 = ctx.Input(kFirstInputIndex);   // 密文 tensor
    Tensor *input1 = ctx.Input(kSecondInputIndex);  // 密钥 tensor
    Tensor *output = ctx.Output(kFirstOutputIndex); // 明文 tensor

    // --- 2. 空数据检查 ---
    // GetDataSize() 返回 tensor 中数据的字节数
    // 如果输入为空，直接返回成功（无需处理）
    if (input0->GetDataSize() == 0 || input1->GetDataSize() == 0) {
        return SUCCESS;
    }

    // --- 3. 获取数据指针 ---
    // GetData() 返回 void*，需要转型为具体类型
    // 【特别注意】此处必须使用 GetDataSize() 获取实际数据长度
    // 绝对不能使用 sizeof(cyphertext) —— 那是指针的大小（8 字节）
    uint8_t *cyphertext = reinterpret_cast<uint8_t *>(input0->GetData());
    uint8_t *key = reinterpret_cast<uint8_t *>(input1->GetData());
    uint8_t *plaintext = reinterpret_cast<uint8_t *>(output->GetData());

    // --- 4. 初始化 ---

    // 4a. 拷贝 S 盒到成员变量
    // SBOX 是匿名命名空间中的 const 数组，在此将其复制到成员变量
    // 以便在子函数中通过成员变量访问
    for (int i = 0; i < 256; i++) {
        sbox_[i] = SBOX[i];
    }

    // 4b. 密钥扩展
    // 将 16 字节原始密钥扩展为 176 字节轮密钥
    keyExpansion(key, roundKeys_);

    // 4c. 设置初始向量和计数器
    // IV 固定为全零，计数器初始值等于 IV
    for (int i = 0; i < AES_BLOCK_BYTES; i++) {
        iv_[i] = iv[i];
        ctr_[i] = iv[i];
    }

    // --- 5. 解密主循环 ---

    // 密钥流缓冲区（16 字节 = 一个 AES 分组大小）
    uint8_t keystream[AES_BLOCK_BYTES];

    // 获取密文数据的实际字节数
    // 【关键修复】之前此处误用了 sizeof(cyphertext)，实际上
    // 应使用 input0->GetDataSize() 获取 tensor 数据的真实大小
    int len = static_cast<int>(input0->GetDataSize());

    int offset = 0;
    while (offset < len) {
        // 步骤 A: 生成密钥流
        // 将当前计数器进行 AES 加密，得到 16 字节密钥流
        // 然后计数器加 1（CTR 模式核心）
        generateKeystream(keystream);

        // 步骤 B: 计算当前分块大小
        // 最后一块可能不足 16 字节，只需处理剩余字节
        int blockSize = (len - offset < AES_BLOCK_BYTES) ?
                        (len - offset) : AES_BLOCK_BYTES;

        // 步骤 C: 解密
        // 明文 = 密文 XOR 密钥流
        // 这是 CTR 模式的核心运算，也是与 ECB/CBC 等模式的主要区别
        for (int i = 0; i < blockSize; i++) {
            plaintext[offset + i] = cyphertext[offset + i] ^ keystream[i];
        }

        // 前进到下一个分块
        offset += blockSize;
    }

    return SUCCESS;
}

// ====================================================================
// keyExpansion: AES-128 密钥扩展
//
// 将 16 字节的原始密钥扩展为 11 个轮密钥（共 176 字节）。
// 算法: 每 4 字节为一组 word，每轮扩展中:
//   1. 对上一个 word 进行 RotWord（循环左移 1 字节）
//   2. 再对每个字节进行 SubWord（查 S 盒）
//   3. 对第一个字节异或 Rcon 轮常量
//   4. 与 AES_KEY_BYTES 前的 word 异或得到新 word
//
// 参数:
//   key:       原始 16 字节密钥
//   roundKeys: 输出的 176 字节扩展密钥缓冲区
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::keyExpansion(
    const uint8_t* key, uint8_t* roundKeys)
{
    // Rcon（轮常量）表
    // Rcon[i] = 2^(i-1) 在 GF(2^8) 中的值
    uint8_t Rcon[11] = {
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10,
        0x20, 0x40, 0x80, 0x1B, 0x36
    };

    // 前 16 字节就是原始密钥
    for (int i = 0; i < AES_KEY_BYTES; i++) {
        roundKeys[i] = key[i];
    }

    int bytesUsed = AES_KEY_BYTES;   // 已填充的轮密钥字节数
    int rconIdx = 1;                  // Rcon 索引
    uint8_t temp[4];                  // 临时存储 4 字节

    while (bytesUsed < AES_EXPANDED_KEY_BYTES) {
        // 取最近生成的 4 字节
        for (int i = 0; i < 4; i++) {
            temp[i] = roundKeys[bytesUsed - 4 + i];
        }

        // 每 16 字节进行一次 RotWord + SubWord + Rcon 变换
        // 这是 AES 密钥扩展的核心步骤
        if (bytesUsed % AES_KEY_BYTES == 0) {
            uint8_t t = temp[0];
            temp[0] = (uint8_t)(SBOX[temp[1]] ^ Rcon[rconIdx]);
            temp[1] = SBOX[temp[2]];
            temp[2] = SBOX[temp[3]];
            temp[3] = SBOX[t];
            rconIdx++;
        }

        // 生成新 4 字节: 与 16 字节前的对应字节异或
        for (int i = 0; i < 4; i++) {
            roundKeys[bytesUsed] = (uint8_t)(
                roundKeys[bytesUsed - AES_KEY_BYTES] ^ temp[i]);
            bytesUsed++;
        }
    }
}

// ====================================================================
// addRoundKey: 轮密钥加
//
// 将 state 与轮密钥进行逐字节异或。
// 这是 AES 每轮加密的最后一个步骤，也是解密时第一轮密钥加。
//
// 参数:
//   state:    当前 16 字节状态
//   roundKey:当前轮密钥（16 字节）
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::addRoundKey(
    uint8_t* state, const uint8_t* roundKey)
{
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

// ====================================================================
// subBytes: 字节代换（非线性替换）
//
// 对 state 的每个字节通过 S 盒查找进行替换。
// S 盒提供了 AES 的混淆特性（Confusion），
// 是算法中唯一的非线性运算。
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::subBytes(uint8_t* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = SBOX[state[i]];
    }
}

// ====================================================================
// shiftRows: 行移位
//
// 将 4x4 的 state 矩阵的各行按不同偏移量循环左移：
//   行 0: 不移位
//   行 1: 左移 1 字节
//   行 2: 左移 2 字节
//   行 3: 左移 3 字节
//
// 注意: AES 的 state 矩阵是按列优先存储的:
//   [0  4  8  12]    (列 0  1  2  3)
//   [1  5  9  13]    (行 0  1  2  3)
//   [2  6  10 14]
//   [3  7  11 15]
// 所以 state[1], state[5], state[9], state[13] 是同一行（第 1 行）
//
// ShiftRows 提供了 AES 的扩散特性（Diffusion），
// 确保每一字节的变更能够扩散到整个分组。
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::shiftRows(uint8_t* state) {
    uint8_t tmp;

    // 第 1 行（索引 1,5,9,13）：左移 1 个字节位置
    tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    // 第 2 行（索引 2,6,10,14）：左移 2 个字节位置
    tmp = state[2];
    uint8_t tmp2 = state[6];
    state[2] = state[10];
    state[6] = state[14];
    state[10] = tmp;
    state[14] = tmp2;

    // 第 3 行（索引 3,7,11,15）：左移 3 个字节位置
    tmp = state[3];
    state[3] = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = tmp;
}

// ====================================================================
// mixColumns: 列混合
//
// 对 state 的每一列在 GF(2^8) 域上进行矩阵乘法混合：
//   new_state[i] = 2*state[i] + 3*state[i+1] + state[i+2] + state[i+3]
//   其中 + 是 XOR，* 是 GF(2^8) 上的乘法
//
// MixColumns 与 ShiftRows 共同提供扩散特性：
//   一个字节的改动会影响到同列的其他 3 个字节。
//
// 使用 xtime 函数实现 GF(2^8) 乘以 2 的运算。
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::mixColumns(uint8_t* state) {
    for (int col = 0; col < 4; col++) {
        int idx = col * 4;
        uint8_t a0 = state[idx + 0];
        uint8_t a1 = state[idx + 1];
        uint8_t a2 = state[idx + 2];
        uint8_t a3 = state[idx + 3];

        // 列混合公式的优化实现
        uint8_t t = a0 ^ a1 ^ a2 ^ a3;
        uint8_t u = a0;

        state[idx + 0] ^= t ^ xtime((uint8_t)(a0 ^ a1));
        state[idx + 1] ^= t ^ xtime((uint8_t)(a1 ^ a2));
        state[idx + 2] ^= t ^ xtime((uint8_t)(a2 ^ a3));
        state[idx + 3] ^= t ^ xtime((uint8_t)(a3 ^ u));
    }
}

// ====================================================================
// encryptBlock: 完整的 AES-128 分组加密
//
// AES-128 加密流程:
//   1. 初始 AddRoundKey
//   2. 第 1~9 轮: SubBytes -> ShiftRows -> MixColumns -> AddRoundKey
//   3. 第 10 轮:  SubBytes -> ShiftRows -> AddRoundKey（无 MixColumns）
//
// 本函数用于 CTR 模式中的密钥流生成：
//   keystream = AES_encrypt(key, counter)
//
// 参数:
//   in:        输入（16 字节计数器）
//   roundKeys: 扩展密钥（176 字节）
//   out:       输出（16 字节密钥流）
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::encryptBlock(
    const uint8_t in[16],
    const uint8_t roundKeys[176],
    uint8_t out[16])
{
    uint8_t state[16];

    // 将输入拷贝到 state
    for (int i = 0; i < AES_KEY_BYTES; i++) {
        state[i] = in[i];
    }

    // 初始轮密钥加
    addRoundKey(state, roundKeys);

    // 第 1 ~ 9 轮（包含 MixColumns）
    for (int r = 1; r < AES_ROUND_NUM; r++) {
        subBytes(state);              // 字节代换（非线性）
        shiftRows(state);             // 行移位（扩散）
        mixColumns(state);            // 列混合（扩散）
        addRoundKey(state, roundKeys + 16 * r);  // 轮密钥加
    }

    // 第 10 轮（最后一轮，不包含 MixColumns）
    subBytes(state);
    shiftRows(state);
    addRoundKey(state, roundKeys + 16 * AES_ROUND_NUM);

    // 输出结果
    for (int i = 0; i < AES_KEY_BYTES; i++) {
        out[i] = state[i];
    }
}

// ====================================================================
// bigEndAdd: CTR 计数器的大端加法
//
// AES-CTR 模式使用大端表示法（Big-Endian）的计数器。
// 每处理完一个 16 字节分组，计数器加 1。
// 大端序意味着最低地址存储最高字节，所以加法从末尾开始。
//
// 例如计数器为 0x00000000000000000000000000000000
// 加 1 后为 0x00000000000000000000000000000001
//
// 参数:
//   ctr:  16 字节计数器（大端序）
//   val:  要加的值（通常为 1）
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::bigEndAdd(uint8_t* ctr, uint32_t val) {
    int carry = 0;
    for (int i = 15; i >= 0; --i) {
        uint32_t sum = (uint32_t)ctr[i] + (val & 0xFF) + carry;
        ctr[i] = (uint8_t)(sum & 0xFF);
        carry = (sum >> 8) & 0xFF;
        val >>= 8;
        if (val == 0 && carry == 0) break;
    }
}

// ====================================================================
// generateKeystream: 生成 16 字节密钥流
//
// CTR 模式的核心运算:
//   keystream[i] = AES_encrypt(key, counter)
//   counter++
//
// 生成的密钥流用于与密文逐字节异或完成解密。
// 由于 AES 加密是确定性的，只要 key 和 counter 相同，
// 加密和解密得到的密钥流完全一致。
// ====================================================================
void AesCtrDecryptAicpuCpuKernel::generateKeystream(uint8_t* keystream) {
    // 对当前计数器值执行 AES 加密，得到密钥流
    encryptBlock(ctr_, roundKeys_, keystream);
    // 计数器递增（CTR 模式要求每个分组的计数器值唯一）
    bigEndAdd(ctr_, 1);
}

// ====================================================================
// REGISTER_CPU_KERNEL: 算子注册宏
//
// 这个宏将算子类型名称与 CpuKernel 子类绑定。
// 宏展开后:
//   1. 定义一个工厂函数 Creator_AES_CTR_DECRYPT_AICPU_Kernel()
//      返回 std::shared_ptr<CpuKernel>
//   2. 调用 RegistCpuKernel() 将算子名与工厂函数注册到全局映射表
//   3. AICPU 框架在收到算子执行请求时，通过算子名查找工厂函数，
//      创建 kernel 实例，然后调用 Compute
//
// 第一个参数 AES_CTR_DECRYPT_AICPU 是算子类型名字符串常量
// 第二个参数 AesCtrDecryptAicpuCpuKernel 是 CpuKernel 子类名
// ====================================================================
REGISTER_CPU_KERNEL(AES_CTR_DECRYPT_AICPU, AesCtrDecryptAicpuCpuKernel);

} // namespace aicpu
