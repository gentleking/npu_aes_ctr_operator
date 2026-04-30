/**
 * Copyright (C)  2020-2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_AES_CTR_DECRYPT_AICPU_H
#define GE_OP_AES_CTR_DECRYPT_AICPU_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(AesCtrDecryptAicpu)
    .INPUT(ciphertext, TensorType({DT_UINT8}))
    .INPUT(key, TensorType({DT_UINT8}))
    .OUTPUT(plaintext, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(AesCtrDecryptAicpu)
}
#endif //GE_OP_AES_CTR_DECRYPT_AICPU_H
