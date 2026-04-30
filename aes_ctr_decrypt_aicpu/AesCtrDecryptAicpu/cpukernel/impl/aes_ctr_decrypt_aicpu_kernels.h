/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:  AI CPU 自定义算子 AesCtrDecryptAicpu 的头文件
 *
 * 算子功能: AES-128 CTR 模式解密
 * 输入0:    密文 (uint8, ND, 任意长度)
 * 输入1:    密钥 (uint8, ND, [16])
 * 输出:     明文 (uint8, ND, 与输入0相同形状)
 *
 * AES-CTR 解密原理:
 *   keystream = AES(key, counter++)   -- 对计数器值进行 AES 加密生成密钥流
 *   plaintext = ciphertext XOR keystream  -- 密文与密钥流异或得到明文
 *
 * 本算子 IV 固定为全零，计数器初始值为 0x00000000000000000000000000000000
 */

#ifndef _AES_CTR_DECRYPT_AICPU_KERNELS_H_
#define _AES_CTR_DECRYPT_AICPU_KERNELS_H_

#include "cpu_kernel.h"    // Ascend CpuKernel 基类，所有 AICPU 算子必须包含
#include "cpu_types.h"     // 公共类型定义（Tensor, CpuKernelContext 等）

// ---- AES-128 算法参数 ----
// AES-128 使用 128 位（16 字节）密钥，128 位（16 字节）分组
// 加密轮数为 10 轮，每轮需要一个 16 字节的轮密钥
static const int AES_KEY_BYTES = 16;               // 密钥长度（字节）
static const int AES_BLOCK_BYTES = 16;              // 分组长度（字节）
static const int AES_ROUND_NUM = 10;                // 加密轮数
static const int AES_EXPANDED_KEY_BYTES = 16 * (AES_ROUND_NUM + 1);  // 扩展密钥总长度 = 176 字节

namespace aicpu {

/*
 * AesCtrDecryptAicpuCpuKernel: AI CPU 核函数类
 *
 * 继承自 CpuKernel，必须实现 Compute 纯虚函数。
 * Compute 由 AICPU 框架在接收到算子执行请求时自动调用。
 *
 * 成员变量:
 *   roundKeys_: 176 字节的扩展密钥（由 keyExpansion 生成）
 *   iv_:        初始向量（固定全零 16 字节）
 *   ctr_:       当前计数器（初始等于 iv，每加密一个分组后 +1）
 *   sbox_:      AES S 盒查找表（256 字节）
 *
 * 静态方法: 不依赖成员变量，仅通过参数传递数据
 * 非静态方法: generateKeystream 访问成员 ctr_ 和 roundKeys_
 */
class AesCtrDecryptAicpuCpuKernel : public CpuKernel {
public:
    ~AesCtrDecryptAicpuCpuKernel() = default;

    /*
     * Compute: AICPU 框架入口函数
     * 参数 ctx: 包含输入输出 tensor、算子属性等上下文信息
     * 返回值:   0 表示成功，非 0 表示失败
     *
     * 调用流程:
     *   Host 调用 aclopExecuteV2 ->
     *   GE 图引擎解析 OM 模型 ->
     *   AICPU 调度器加载 kernel SO ->
     *   调用 REGISTER_CPU_KERNEL 注册的工厂函数创建实例 ->
     *   调用 Compute
     */
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    // ========== AES 核心算法 ==========
    // 这些函数设计为 static，不依赖成员变量，便于单元测试

    /*
     * keyExpansion: AES-128 密钥扩展算法
     * 将 16 字节原始密钥扩展为 176 字节轮密钥
     * 算法: 每 4 字节一组进行 RotWord -> SubWord -> Rcon 变换
     */
    static void keyExpansion(const uint8_t* key, uint8_t* roundKeys);

    /* addRoundKey: 轮密钥加
     * state 与轮密钥进行逐字节异或 */
    static void addRoundKey(uint8_t* state, const uint8_t* roundKey);

    /* subBytes: 字节代换
     * 对 state 的每个字节通过 S 盒查找替换为非线性的值 */
    static void subBytes(uint8_t* state);

    /* shiftRows: 行移位
     * 将 4x4 state 矩阵的各行按不同偏移量循环左移 */
    static void shiftRows(uint8_t* state);

    /* mixColumns: 列混合
     * 对 state 的各列在 GF(2^8) 域上进行矩阵乘法混合 */
    static void mixColumns(uint8_t* state);

    /* encryptBlock: 完整 AES-128 加密一个分组
     * 包括: AddRoundKey -> (SubBytes+ShiftRows+MixColumns+AddRoundKey) x9 轮 ->
     *       SubBytes+ShiftRows+AddRoundKey
     */
    static void encryptBlock(const uint8_t in[16], const uint8_t roundKeys[176], uint8_t out[16]);

    // ========== CTR 模式辅助 ==========

    /* bigEndAdd: 大端计数器加法
     * 对大端表示的 16 字节计数器加 val
     * 用于 CTR 模式每处理一个分组后计数器递增 */
    static void bigEndAdd(uint8_t* ctr, uint32_t val);

    /* generateKeystream: 生成密钥流
     * 将当前计数器值经 AES 加密后得到 16 字节密钥流
     * 然后计数器加 1 */
    void generateKeystream(uint8_t* keystream);

    // ========== 成员变量 ==========
    // 这些变量在 Compute 的每次调用中初始化

    uint8_t roundKeys_[AES_EXPANDED_KEY_BYTES];   // 扩展密钥 (176 字节)
    uint8_t iv_[AES_BLOCK_BYTES];                 // 初始向量 IV (16 字节，固定全零)
    uint8_t ctr_[AES_BLOCK_BYTES];                // 当前计数器 (16 字节，大端)
    uint8_t sbox_[256];                           // AES S 盒查找表 (256 字节)
};

} // namespace aicpu
#endif
