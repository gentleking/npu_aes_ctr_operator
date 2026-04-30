#include "aes_ctr_decrypt_aicpu.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(AesCtrDecryptAicpuInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AesCtrDecryptAicpu, AesCtrDecryptAicpuVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AesCtrDecryptAicpu, AesCtrDecryptAicpuInferShape);
VERIFY_FUNC_REG(AesCtrDecryptAicpu, AesCtrDecryptAicpuVerify);

}  // namespace ge
