#ifndef AES_CTR_CUSTOM_TILING_H
#define AES_CTR_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// TilingData结构用于Host向Device传递全局信息和配置参数
BEGIN_TILING_DATA_DEF(AesCtrCustomTilingData)
  // 全局明文总长度（单位：byte），由Host TilingFunc填充
  TILING_DATA_FIELD_DEF(uint32_t, global_total_bytes);
  // 用户通过属性配置的单个流水线tile的大小（单位：byte）
  TILING_DATA_FIELD_DEF(uint32_t, config_pipeline_tile_bytes);
END_TILING_DATA_DEF;

// 注册TilingData结构到AesCtrCustom算子
REGISTER_TILING_DATA_CLASS(AesCtrCustom, AesCtrCustomTilingData)

} // namespace optiling
#endif // AES_CTR_CUSTOM_TILING_H
