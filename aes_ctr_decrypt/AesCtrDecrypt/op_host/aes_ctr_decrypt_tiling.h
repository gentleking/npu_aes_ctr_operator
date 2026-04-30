
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AesCtrDecryptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  // TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AesCtrDecrypt, AesCtrDecryptTilingData)
}
