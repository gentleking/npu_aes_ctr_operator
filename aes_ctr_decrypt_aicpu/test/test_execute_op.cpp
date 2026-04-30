#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "acl/acl_op.h"

int main() {
    if (aclInit("scripts/acl.json") != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl; return 1;
    }
    if (aclopSetModelDir("op_models") != ACL_SUCCESS) {
        std::cerr << "aclopSetModelDir failed" << std::endl; return 1;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl; return 1;
    }

    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);
    bool isDevice = (runMode == ACL_DEVICE);
    std::cout << "isDevice=" << isDevice << std::endl;

    // Try querying the operator
    aclopAttr *attr = aclopCreateAttr();
    if (attr == nullptr) {
        std::cout << "aclopCreateAttr returned null" << std::endl;
    } else {
        std::cout << "aclopCreateAttr ok" << std::endl;
        aclopDestroyAttr(attr);
    }

    // Check if the operator exists in the loaded models
    int isExist = aclopExist("AesCtrDecryptAicpu");
    std::cout << "aclopExist(AesCtrDecryptAicpu) = " << isExist << std::endl;
    isExist = aclopExist("NonExistentOp");
    std::cout << "aclopExist(NonExistentOp) = " << isExist << std::endl;

    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
