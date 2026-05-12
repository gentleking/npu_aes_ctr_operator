/**
 * @file aes_ctr_decrypt_tiling.h
 * @brief Tiling 数据结构定义
 *
 * Tiling 是 Ascend 算子开发中的关键概念，用于在 Host 侧（CPU）将
 * 算子执行的参数提前计算好，传递给 Device 侧（NPU）的 kernel 使用。
 *
 * 为什么需要 Tiling？
 *   NPU 的 Local Memory（LM）大小有限（一般为几十KB到几百KB），
 *   无法一次性容纳全部输入数据。Tiling 将大张量切分成多个小 tile，
 *   让 kernel 能够以流水线方式分块处理。
 *
 * 本算子的 Tiling 传递两个参数：
 *   - totalLength: 输入数据的 uint8 元素总数（即字节数），用于
 *     kernel 侧计算总工作量
 *   - tileNum: tile 数量，kernel 侧结合 TILE_LENGTH（64）计算
 *     流水线迭代次数
 *
 * 宏展开说明：
 *   BEGIN_TILING_DATA_DEF / END_TILING_DATA_DEF
 *     定义了一个包含指定字段的 tiling 数据结构体，自动生成序列化/
 *     反序列化方法。
 *   TILING_DATA_FIELD_DEF(type, name)
 *     声明一个 tiling 字段。
 *   REGISTER_TILING_DATA_CLASS
 *     将 tiling 数据结构注册到算子的 tiling 配置中。
 */

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AesCtrDecryptTilingData)
  // totalLength: 输入 cyphertext 张量的元素总数（uint8 元素个数 = 字节数）
  // kernel 侧将其除以 2 得到 half 单元数，再按核数分块
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  // tileNum: 每个核处理的 tile 数量建议值
  // kernel 侧用此值结合 BUFFER_NUM=2 计算实际的 TILE_LENGTH=64
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AesCtrDecrypt, AesCtrDecryptTilingData)

}
