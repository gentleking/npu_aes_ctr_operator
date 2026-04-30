/**
 * @file AesCtrEncryptKernelNpuApi.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

// 对外接口头文件，包含op_plugin所有ACLNN算子对外的函数原型
#include "op_plugin/AclOpsInterface.h"
// 引用 ACLOP 算子声明头文件
#include "op_plugin/OpApiInterface.h"
// torch调用ACLNN算子时，所依赖的基础函数对应的头文件
#include "op_plugin/utils/op_api_common.h"

// ACLNN算子定义在op_api命名空间中
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_aes_ctr_decrypt(const at::Tensor &cyphertext, const at::Tensor &key)
{
    at::Tensor plaintext = npu_preparation::apply_tensor_without_format(cyphertext); // Create output memory

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAesCtrDecrypt, cyphertext, key, plaintext);
    return plaintext;
}
} // namespace op_api
