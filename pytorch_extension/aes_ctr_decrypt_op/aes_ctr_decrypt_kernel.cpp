#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclnn_aes_ctr_decrypt.h"

#define ACL_CHECK(expr)                          \
    do {                                         \
        aclnnStatus _ret = (expr);               \
        TORCH_CHECK(_ret == ACL_SUCCESS,          \
            "ACL error: ", _ret, " at ",         \
            __FILE__, ":", __LINE__);             \
    } while (0)

static aclDataType toAclDataType(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Byte: return ACL_UINT8;
        case at::ScalarType::Char: return ACL_INT8;
        case at::ScalarType::Int:  return ACL_INT32;
        case at::ScalarType::Long: return ACL_INT64;
        case at::ScalarType::Half: return ACL_FLOAT16;
        case at::ScalarType::Float: return ACL_FLOAT;
        default: TORCH_CHECK(false, "unsupported dtype: ", dtype);
    }
}

static aclTensor* wrapTensor(const at::Tensor& t) {
    auto sizes = t.sizes();
    auto ndim = sizes.size();
    return aclCreateTensor(
        sizes.data(), ndim, toAclDataType(t.scalar_type()),
        nullptr, 0, ACL_FORMAT_ND,
        sizes.data(), ndim, t.data_ptr());
}

at::Tensor npu_aes_ctr_decrypt_kernel(const at::Tensor& cyphertext, const at::Tensor& key) {
    TORCH_CHECK(cyphertext.device().type() == c10::DeviceType::PrivateUse1,
                "cyphertext must be on NPU");
    TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1,
                "key must be on NPU");

    at::Tensor out = at::empty_like(cyphertext, c10::DeviceType::PrivateUse1);

    aclTensor* cAcl = wrapTensor(cyphertext);
    aclTensor* kAcl = wrapTensor(key);
    aclTensor* oAcl = wrapTensor(out);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    size_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnAesCtrDecryptGetWorkspaceSize(cAcl, kAcl, oAcl, &wsSize, &executor));

    void* ws = nullptr;
    if (wsSize > 0) {
        ACL_CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(aclnnAesCtrDecrypt(ws, wsSize, executor, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    if (ws) aclrtFree(ws);
    aclDestroyTensor(cAcl);
    aclDestroyTensor(kAcl);
    aclDestroyTensor(oAcl);

    return out;
}

TORCH_LIBRARY(aes_ctr, m) {
    m.def("decrypt(Tensor cyphertext, Tensor key) -> Tensor");
    m.impl("decrypt", npu_aes_ctr_decrypt_kernel);
}
