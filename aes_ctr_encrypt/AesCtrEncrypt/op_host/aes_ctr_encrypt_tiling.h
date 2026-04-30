
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AesCtrEncryptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AesCtrEncrypt, AesCtrEncryptTilingData)
}
