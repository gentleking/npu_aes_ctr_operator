/**
 * @file aes_ctr_decrypt.cpp
 * @brief Ascend C 自定义 AES-CTR 解密算子 (AI-CORE 实现)
 *
 * ============ NPU 算子开发的核心概念 ============
 *
 * 1. 两级存储架构
 *    - GM (Global Memory): 大容量，高延迟，所有核共享
 *      类似 CPU 的 RAM，但位于 NPU 芯片外部（HBM）
 *    - Local Memory: 小容量（几十KB/核），低延迟，每个核独享
 *      kernel 必须将数据从 GM 搬到 Local Memory 后才能计算
 *      本算子的 TILE_LENGTH=1024 halfs(2048 字节) 即为 Local Memory 的 tile 大小
 *
 * 2. TQue 流水线（三阶段，双缓冲区）
 *    为了隐藏 GM → Local 的数据搬运延迟，采用 CopyIn → Compute → CopyOut
 *    三阶段流水线并行：
 *      CopyIn（搬入）:    GM → Local（DataCopy 指令）
 *      Compute（计算）:   Local 中执行 AES-CTR 解密
 *      CopyOut（搬出）:   Local → GM（DataCopy 指令）
 *    缓冲数量 BUFFER_NUM=2 实现双缓冲：一块 buffer 在计算时，
 *    另一块 buffer 在搬运数据，两者并行。
 *
 * 3. half 类型 reinterpretation
 *    Ascend C 的 LocalTensor<T> 支持多种数据类型，但标量运算使用 half。
 *    输入是 uint8 但 kernel 用 half* 访问，因为：
 *    - 2 个 uint8 字节打包成 1 个 half（16 位）传输
 *    - Compute 阶段将 half 拆回 uint8[] 做 AES 计算
 *    - 计算完再打包回 half 写回 GM
 *    因此数据总字节数必须为偶数。
 *
 * 4. 多核并行
 *    AI Core 有 8 个物理核（在 310P3 上为 8 核），每个核处理一部分数据。
 *    数据在 Init() 中按核数均分，每个核通过 GetBlockIdx() 获取自己的编号。
 *    计数器（IV）按核的起始位置做偏移，保证各核使用不同的密钥流。
 *
 * 5. tiling 参数传递
 *    Host 侧 TilingFunc 计算好的参数（totalLength, tileNum）通过
 *    GET_TILING_DATA 宏在 kernel 入口反序列化，指导 Init() 做分块。
 *
 * ===============================================
 */

#include "kernel_operator.h"
#include <cstring>

// ==================== 编译期常量 ====================
constexpr int32_t BUFFER_NUM = 2;       // 双缓冲
constexpr int32_t TILE_LENGTH = 1024;    // 每个 tile 的 half 数量（优化：从64增至1024）
                                         // 用于分配 Local Memory buffer
                                         // 对应 1024 * 2 = 2048 字节/ tile

// ==================== AES 参数定义 ====================
static const int AES_KEY_BYTES = 16;              // AES-128 密钥字节数
static const int AES_BLOCK_BYTES = 16;            // AES 分组大小（128 位）
static const int AES_ROUND_NUM = 10;              // AES 轮数
static const int AES_EXPANDED_KEY_BYTES = 16 * (AES_ROUND_NUM + 1);  // 扩展密钥长度 176 字节


// ==================== AES 算法核心函数 ====================
// 函数被 __aicore__ 修饰，表示在 NPU AI Core 上执行
// 这些函数在 KernelAesCtrDecrypt::Compute() 中被调用，
// 操作 Local Memory 中的数据缓冲区（plainBuf / cipherBuf）

/**
 * @brief GF(2^8) 中左移操作
 *        AES MixColumns 的辅助函数
 *        若输入最高位为 1，则左移后异或 0x1B（不可约多项式）
 * @param x  输入字节
 * @return  变换后的字节
 */
__aicore__ inline static uint8_t xtime(uint8_t x)
{
    return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00));
}

/**
 * @brief AES-128 密钥扩展
 *        将原始 16 字节密钥通过密钥调度算法扩展为 176 字节（11 轮子密钥）
 *
 * 密钥扩展是 AES 算法的前置步骤，每轮加密需要不同的 16 字节子密钥。
 * 扩展过程：每 4 字节为一组，每 16 字节做一次 RotWord + SubWord + Rcon 变换。
 *
 * @param key       原始密钥，16 字节
 * @param roundKeys 输出缓冲区，176 字节
 * @param sbox      AES S-box 查找表（256 字节）
 */
static __aicore__ void KeyExpansion(const uint8_t *key, uint8_t *roundKeys, const uint8_t *sbox)
{
    uint8_t Rcon[11] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
    // 复制原始密钥（前 16 字节）
    for (int i = 0; i < AES_KEY_BYTES; i++) {
        roundKeys[i] = key[i];
    }

    int bytesUsed = AES_KEY_BYTES;  // 已生成的字节数
    int rconIdx = 1;
    volatile uint8_t temp[4];

    while (bytesUsed < AES_EXPANDED_KEY_BYTES) {
        for (int i = 0; i < 4; i++) {
            temp[i] = roundKeys[bytesUsed - 4 + i];
        }
        // 每 16 字节做一次轮变换
        if ((bytesUsed % AES_KEY_BYTES) == 0) {
            uint8_t t = temp[0];
            temp[0] = (uint8_t)(sbox[temp[1]] ^ Rcon[rconIdx]);
            temp[1] = sbox[temp[2]];
            temp[2] = sbox[temp[3]];
            temp[3] = sbox[t];
            rconIdx++;
        }
        // 生成新的 4 字节子密钥
        for (int i = 0; i < 4; i++) {
            roundKeys[bytesUsed] = (uint8_t)(roundKeys[bytesUsed - AES_KEY_BYTES] ^ temp[i]);
            bytesUsed++;
        }
    }
}

/**
 * @brief AddRoundKey：状态与子密钥逐字节异或
 */
static __aicore__ inline void AddRoundKey(uint8_t *state, const uint8_t *roundKey)
{
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/**
 * @brief 前 9 轮变换：SubBytes → ShiftRows → MixColumns
 */
static __aicore__ void SubShiftMix(uint8_t *state, uint32_t blockId, const uint8_t *sbox)
{
    // SubBytes：用 S-box 替换每个字节
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }

    // ShiftRows：行移位
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

    // MixColumns：列混合（用 xtime 实现 GF(2^8) 乘法）
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
 * @brief 最后一轮变换：SubBytes → ShiftRows（无 MixColumns）
 */
static __aicore__ void SubShift(uint8_t *state, const uint8_t *sbox)
{
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }
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
 * @brief 单块 AES-128 加密
 *
 * 标准 AES-128 加密流程：
 *   初始轮 (AddRoundKey)
 *   → 第 1-9 轮 (SubBytes → ShiftRows → MixColumns → AddRoundKey)
 *   → 第 10 轮 (SubBytes → ShiftRows → AddRoundKey, 无 MixColumns)
 *
 * @param in         输入（16 字节）
 * @param roundKeys  扩展密钥（176 字节）
 * @param out        输出密文（16 字节）
 */
static __aicore__ void AesEncryptBlock(const uint8_t in[16],
    const uint8_t roundKeys[AES_EXPANDED_KEY_BYTES],
    uint8_t out[16], uint32_t blockId, const uint8_t *sbox)
{
    uint8_t state[16];
    for (int i = 0; i < 16; i++) state[i] = in[i];

    AddRoundKey(state, roundKeys);
    for (int r = 1; r < AES_ROUND_NUM; r++) {
        SubShiftMix(state, blockId, sbox);
        AddRoundKey(state, roundKeys + 16 * r);
    }
    SubShift(state, sbox);
    AddRoundKey(state, roundKeys + 16 * AES_ROUND_NUM);

    for (int i = 0; i < 16; i++) out[i] = state[i];
}

/**
 * @brief 大端加法
 *        将 val 加到表示为大端格式的 128 位计数器 ctr 上
 *        AES-CTR 的计数器使用大端计数
 */
__aicore__ inline void BigEndAdd(uint8_t *ctr, uint32_t val)
{
    int carry = 0;
    for (int i = 15; i >= 0; --i) {
        uint32_t sum = (uint32_t)ctr[i] + (val & 0xFF) + carry;
        ctr[i] = (uint8_t)(sum & 0xFF);
        carry = (sum >> 8) & 0xFF;
        val >>= 8;
        if (val == 0 && carry == 0) break;
    }
}

/**
 * @brief AES-CTR 模式加密/解密
 *
 * CTR 模式将 AES 分组加密转为流加密：
 *   密钥流 = AES(key, counter)
 *   密文 = 明文 XOR 密钥流
 *   counter 每处理一个 16 字节块后自增
 *
 * AES-CTR 加解密使用相同的操作（AES 加密），区别仅在于输入。
 * 本算子名为 decrypt，实际调用 AesCtrEncrypt 做解密（密文 XOR 密钥流 = 明文）。
 *
 * @param plaintext   输入（密文）
 * @param ciphertext  输出（明文）
 * @param lengthBytes 数据字节数
 * @param roundKeys   扩展密钥
 * @param iv          初始向量/计数器
 */
static __aicore__ void AesCtrEncrypt(const uint8_t *plaintext, uint8_t *ciphertext,
    int32_t lengthBytes, const uint8_t *roundKeys, uint8_t iv[16],
    int32_t blockId, const uint8_t *sbox)
{
    uint8_t ctrBlock[16];
    for (int i = 0; i < 16; i++) ctrBlock[i] = iv[i];

    uint8_t keystream[16];
    int offset = 0;
    while (offset < lengthBytes) {
        // 1. 用 AES 加密当前计数器值，生成 16 字节密钥流
        AesEncryptBlock(ctrBlock, roundKeys, keystream, blockId, sbox);
        // 2. 密文 XOR 密钥流 → 明文（最后一块可能不足 16 字节）
        int blockSize = (lengthBytes - offset < 16) ? (lengthBytes - offset) : 16;
        for (int i = 0; i < blockSize; i++) {
            ciphertext[offset + i] = plaintext[offset + i] ^ keystream[i];
        }
        offset += blockSize;
        // 3. 计数器自增（大端）
        for (int i = 15; i >= 0; i--) {
            ctrBlock[i]++;
            if (ctrBlock[i] != 0) break;
        }
    }
    for (int i = 0; i < 16; i++) iv[i] = ctrBlock[i];
}


#ifndef ASCENDC_CPU_DEBUG

// ==================== NPU 模式：TQue 流水线实现 ====================

/**
 * @class KernelAesCtrDecrypt
 * @brief AI-CORE kernel 主类，实现三阶段流水线解密
 *
 * 【流水线原理】
 * NPU 的 Local Memory 很小，不能一次处理全部数据。
 * 本实现将每个核的任务切成多个小 tile，用 CopyIn → Compute → CopyOut
 * 三阶段流水线并行处理。
 *
 * 流水线时序（以 BUFFER_NUM=2 为例）：
 *   Time 1: CopyIn(tile0)          — 数据从 GM 搬到 Local(LM)
 *   Time 2: Compute(tile0)         — LM 中计算 AES 解密
 *           CopyIn(tile1)          — 同时搬入下一块数据（乒乓）
 *   Time 3: CopyOut(tile0)         — 结果从 LM 写回 GM
 *           Compute(tile1)         — 同时计算下一块
 *           CopyIn(tile2)          — 同时搬入再下一块
 *   ...
 *
 * 【half 类型说明】
 * Ascend C 的 LocalTensor 通常使用 half（16 位浮点）作为基本运算单元。
 * 输入数据是 uint8（8 位），为了能在 LocalTensor 中存储，将 2 个 uint8
 * 打包成 1 个 half。计算时再拆回 uint8 做 AES 运算。
 * 这种"打包-计算-拆包"是 NPU 开发的常见模式。
 *
 * 【多核并行】
 * 每个 AI Core 通过 GetBlockIdx() 获取自己的核编号（0-7 共 8 核）。
 * 数据在 Init() 中按核均分，每个核处理一部分连续的 GM 数据。
 * 计数器（IV）按核对应的数据块起始位置做偏移，确保各核使用互不重叠的
 * 密钥流——这是 CTR 模式多核并行的关键。
 */
class KernelAesCtrDecrypt {
public:
    /**
     * @brief 构造函数：初始化 AES S-box
     *
     * S-box 是 AES SubBytes 步骤的查找表，256 字节常数。
     * 在 NPU 上，S-box 存储在 kernel 的 Local Memory 中。
     */
    __aicore__ inline KernelAesCtrDecrypt()
    {
        // AES 标准 S-box（常量，来自 FIPS 197）
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
        for (int i = 0; i < 256; i++) {
            sbox[i] = sbox_init[i];
        }
    }

    /**
     * @brief Init：初始化 GM 缓冲区、流水线、密钥扩展和 IV 偏移
     *
     * 本函数执行以下操作：
     *
     * 1. 动态分块
     *    根据 tiling 传入的 totalLength 动态计算。totalLength 是 
     *    uint8 元素数（=字节数），除以 2 得 half 单元数，再按核数均分。
     *
     * 2. 设置 GlobalTensor 的 GM 地址和长度
     *    每个核通过 startOffset 定位到自己要处理的那段 GM 数据
     *
     * 3. 初始化 TQue 流水线缓冲区
     *    分配 BUFFER_NUM 个 Local Memory buffer，每个 TILE_LENGTH 个 half
     *
     * 4. 密钥扩展
     *    从 GM 读取原始密钥，在本地做 KeyExpansion
     *
     * 5. IV 偏移
     *    每个核的计数器从不同的起始位置开始，保证各核使用不同的密钥流
     *    偏移量 = startOffset / 8（因为每组 16 字节 = 8 个 half）
     *
     * @param plaintext   GM 地址：输出（解密后的明文数据）
     * @param key          GM 地址：密钥（16 字节）
     * @param ciphertext   GM 地址：输入（待解密的密文数据）
     * @param totalLength  uint8 元素总数（来自 tiling，= 字节数）
     * @param tileNum      tile 数量建议值（来自 tiling，固定为 8）
     */
    __aicore__ inline void Init(GM_ADDR plaintext, GM_ADDR key, GM_ADDR ciphertext,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // ---- 动态分块计算 ----
        // blockDim: 当前运行的 AI Core 数量（Host 侧 TilingFunc 中 SetBlockDim 设置）
        int32_t blockDim = AscendC::GetBlockNum();
        // totalHalfs: totalLength（uint8 元素数）转为 half 单元数
        // 因为 kernel 内部用 half* 访问 GM 数据，每 2 个 uint8 = 1 个 half
        int32_t totalHalfs = static_cast<int32_t>(totalLength / 2);
        int32_t blockIdx = AscendC::GetBlockIdx();

        // 数据均匀分配到各核，余数分配给前几个核
        int32_t baseBlockLen = totalHalfs / blockDim;
        int32_t rem = totalHalfs % blockDim;
        // startOffset: 本核在 GM 中的起始偏移（half 单位）
        int32_t startOffset = blockIdx * baseBlockLen + (blockIdx < rem ? blockIdx : rem);
        // thisBlockLen_: 本核要处理的 half 数量
        thisBlockLen_ = baseBlockLen + (blockIdx < rem ? 1 : 0);

        // 如果本核没有数据（理论上不会发生，但做防御）
        if (thisBlockLen_ <= 0) {
            thisBlockLen_ = 0;
            loopCount_ = 0;
            return;
        }

        // 计算本次 kernel 调用需要的 tile 迭代次数
        // 每个 tile 处理 TILE_LENGTH 个 half（64 half = 128 字节）
        loopCount_ = (thisBlockLen_ + TILE_LENGTH - 1) / TILE_LENGTH;

        // ---- 设置 GM buffer ----
        // 将 GM 地址转换为 half 指针，偏移到本核的数据起始位置
        // plaintext 和 ciphertext 在 GM 中是 uint8 数组，kernel 用 half 指针访问
        plaintextGm.SetGlobalBuffer((__gm__ half *)plaintext + startOffset, thisBlockLen_);
        ciphertextGm.SetGlobalBuffer((__gm__ half *)ciphertext + startOffset, thisBlockLen_);

        // ---- 初始化 TQue 流水线 ----
        // 分配 BLOCK_NUM=2 个 buffer，每个 TILE_LENGTH * sizeof(half) = 64*2 = 128 字节
        // inQueuePlain:  输入队列，存放从 GM 搬入的密文数据
        // outQueueCipher: 输出队列，存放解密后的明文数据
        pipe.InitBuffer(inQueuePlain, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueCipher, BUFFER_NUM, TILE_LENGTH * sizeof(half));

        // ---- 密钥复制 + 扩展 ----
        // 从 GM 读取 16 字节密钥到 Local Memory（localKey）
        uint8_t localKey[AES_KEY_BYTES];
        volatile __gm__ uint8_t *gm_key = key;
        for (int i = 0; i < AES_KEY_BYTES; i++) {
            localKey[i] = gm_key[i];
        }
        // 初始化 IV 为全零（算子规格约定：使用全零 IV）
        for (int i = 0; i < AES_BLOCK_BYTES; i++) {
            IV[i] = 0;
        }
        // 执行 AES-128 密钥扩展（在 Local Memory 中完成）
        KeyExpansion(localKey, expandedKey, sbox);

        // ---- IV 偏移（多核并行关键） ----
        // 每个核的计数器起始位置不同，保证密钥流不重叠
        // startOffset 是 half 单位，转为 AES 块数：(startOffset * 2 字节) / 16 字节 = startOffset / 8
        uint32_t offsetBlocks = static_cast<uint32_t>(startOffset / 8);
        uint8_t tmpIV[16];
        for (int i = 0; i < 16; i++) {
            tmpIV[i] = IV[i];
        }
        BigEndAdd(tmpIV, offsetBlocks);
        for (int i = 0; i < 16; i++) {
            IV[i] = tmpIV[i];
        }
    }

    /**
     * @brief Process：流水线主循环
     *
     * 每个迭代执行一次完整的 CopyIn → Compute → CopyOut。
     * loopCount_ 由 Init() 根据数据量动态计算。
     * 如果当前核没有数据（thisBlockLen_ <= 0），直接返回。
     */
    __aicore__ inline void Process()
    {
        if (thisBlockLen_ <= 0) return;
        for (int i = 0; i < loopCount_; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /**
     * @brief 获取第 idx 个 tile 的实际长度
     *        最后一个 tile 可能不足 TILE_LENGTH
     */
    __aicore__ inline int32_t GetTileLen(int32_t idx)
    {
        int32_t processed = idx * TILE_LENGTH;
        int32_t remaining = thisBlockLen_ - processed;
        return (remaining > TILE_LENGTH) ? TILE_LENGTH : (remaining > 0 ? remaining : TILE_LENGTH);
    }

    /**
     * @brief Stage 1: CopyIn — 数据从 GM 搬到 Local Tensor
     *
     * 使用 DataCopy 指令将 GM 中的密文数据复制到 Local Tensor。
     * TQue 的 AllocTensor/EnQue 机制负责 Local Memory 管理。
     *
     * NPU 特性：DataCopy 是异步 DMA 操作，由硬件自动完成，
     * 不会阻塞 pipeline 中的其他阶段。
     *
     * @param idx tile 索引
     */
    __aicore__ inline void CopyIn(int idx)
    {
        int32_t tileLen = GetTileLen(idx);
        AscendC::LocalTensor<half> plainLocal = inQueuePlain.AllocTensor<half>();
        AscendC::DataCopy(plainLocal, plaintextGm[idx * TILE_LENGTH], tileLen);
        inQueuePlain.EnQue(plainLocal);
    }

    /**
     * @brief Stage 2: Compute — 在 Local Tensor 中执行 AES-CTR 解密
     *
     * 核心计算流程：
     *   1. 从 inQueuePlain 出队密文 LocalTensor
     *   2. half → uint8[] 转换（2 字节解包）
     *   3. 调用 AesCtrEncrypt 执行 AES-CTR 解密
     *   4. uint8[] → half 转换（2 字节打包）
     *   5. 将结果入队到 outQueueCipher
     *
     * @param idx tile 索引
     */
    __aicore__ inline void Compute(int idx)
    {
        int32_t blockIdx = AscendC::GetBlockIdx();
        AscendC::LocalTensor<half> plainLocal = inQueuePlain.DeQue<half>();
        AscendC::LocalTensor<half> cipherLocal = outQueueCipher.AllocTensor<half>();

        int32_t tileLen = GetTileLen(idx);
        int32_t tileBytes = tileLen * 2;  // half 数 → 字节数

        // 复制当前 IV 到临时缓冲区（AesCtrEncrypt 会修改 IV）
        uint8_t TmpIv[16];
        for (int i = 0; i < 16; i++) {
            TmpIv[i] = IV[i];
        }

        // half → uint8[] 转换
        // plainLocal 中存的是 half（16 位），需要拆成 uint8 才能做 AES 运算
        constexpr int maxTileBytes = TILE_LENGTH * 2;
        uint8_t plainBuf[maxTileBytes];
        for (int i = 0; i < tileLen; i++) {
            half hVal = plainLocal.GetValue(i);
            uint16_t halfBits = *reinterpret_cast<uint16_t *>(&hVal);
            plainBuf[2 * i + 0] = static_cast<uint8_t>(halfBits & 0xFF);
            plainBuf[2 * i + 1] = static_cast<uint8_t>((halfBits >> 8) & 0xFF);
        }

        // 执行 AES-CTR 解密（实际使用 AesCtrEncrypt，CTR 加解密相同）
        uint8_t cipherBuf[maxTileBytes];
        AesCtrEncrypt(plainBuf, cipherBuf, tileBytes, expandedKey, TmpIv, blockIdx, sbox);

        // 更新本地 IV
        for (int i = 0; i < 16; i++) {
            IV[i] = TmpIv[i];
        }

        // uint8[] → half 转换
        for (int i = 0; i < tileLen; i++) {
            uint16_t halfBits = static_cast<uint16_t>(cipherBuf[2 * i]) |
                                (static_cast<uint16_t>(cipherBuf[2 * i + 1]) << 8);
            half val = *reinterpret_cast<half *>(&halfBits);
            cipherLocal.SetValue(i, val);
        }

        outQueueCipher.EnQue<half>(cipherLocal);
        inQueuePlain.FreeTensor(plainLocal);
    }

    /**
     * @brief Stage 3: CopyOut — 结果从 Local Tensor 写回 GM
     *
     * 将解密后的明文数据通过 DataCopy 从 Local Memory 写回 GM。
     *
     * @param idx tile 索引
     */
    __aicore__ inline void CopyOut(int idx)
    {
        int32_t tileLen = GetTileLen(idx);
        AscendC::LocalTensor<half> cipherLocal = outQueueCipher.DeQue<half>();
        AscendC::DataCopy(ciphertextGm[idx * TILE_LENGTH], cipherLocal, tileLen);
        outQueueCipher.FreeTensor(cipherLocal);
    }

private:
    // ---- 流水线核心组件 ----
    AscendC::TPipe pipe;  // Pipe 管理器，控制 Local Memory 分配和 pipeline 调度
    // TQue: 带位置的模板队列，QuePosition::VECIN/VECOUT 指定数据存放位置
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePlain;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueCipher;

    // ---- GM 访问句柄 ----
    AscendC::GlobalTensor<half> plaintextGm;   // 输入密文（GM 地址）
    AscendC::GlobalTensor<half> ciphertextGm;   // 输出明文（GM 地址）

    // ---- 本地存储的 AES 状态 ----
    uint8_t IV[AES_BLOCK_BYTES];                // 计数器/初始向量（16 字节）
    uint8_t expandedKey[AES_EXPANDED_KEY_BYTES]; // 扩展密钥（176 字节）
    uint8_t sbox[256];                          // AES S-box

    // ---- 动态分块状态 ----
    int32_t thisBlockLen_;  // 当前核处理的 half 数量（由 Init 动态计算）
    int32_t loopCount_;     // 流水线迭代次数（由 Init 动态计算）
};


/**
 * @brief Kernel 入口函数
 *
 * 这是 NPU 执行 kernel 的入口点，由 CANN 框架调用。
 * 使用 extern "C" __global__ __aicore__ 修饰，指定：
 *   - extern "C": 避免 C++ name mangling，让框架能找到符号
 *   - __global__: 表示这是一个 NPU kernel 入口
 *   - __aicore__: 在 AI Core 上执行
 *
 * GET_TILING_DATA 宏从 tiling buffer 中反序列化 Host 侧传过来的参数
 *
 * 5000 行前）会有大块时间消耗。
 *
 * @param cyphertext  输入 GM 地址：密文
 * @param key         输入 GM 地址：密钥
 * @param plaintext   输出 GM 地址：明文
 * @param workspace   工作空间 GM 地址（未使用）
 * @param tiling      tiling 数据 GM 地址
 */
extern "C" __global__ __aicore__ void aes_ctr_decrypt(GM_ADDR cyphertext, GM_ADDR key,
    GM_ADDR plaintext, GM_ADDR workspace, GM_ADDR tiling)
{
    // 从 tiling buffer 反序列化 Host 侧传过来的参数
    // tiling_data.totalLength: uint8 元素总数
    // tiling_data.tileNum: tile 数量建议值
    GET_TILING_DATA(tiling_data, tiling);

    KernelAesCtrDecrypt op;
    op.Init(cyphertext, key, plaintext, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

/**
 * @brief CPU 调试模式的入口（仅用于 DS9 仿真调试）
 *
 * 在非 CPU_DEBUG 模式下，这个函数不会被编译执行。
 * 实际运行时直接调用上面的 aes_ctr_decrypt kernel。
 */
void aes_ctr_decrypt_do(uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *ciphertext, uint8_t *key, uint8_t *plaintext,
    uint8_t *workspace, uint8_t *tiling)
{
    aes_ctr_decrypt<<<blockDim, l2ctrl, stream>>>(ciphertext, key, plaintext);
}

#endif
