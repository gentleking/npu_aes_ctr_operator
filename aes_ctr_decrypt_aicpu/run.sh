#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
AICPU_DIR=$PROJECT_DIR/AesCtrDecryptAicpu
TEST_DIR=$PROJECT_DIR/test
BUILD_DIR=$AICPU_DIR/build_out

SOC_VERSION=Ascend310P3
DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
TOOLCHAIN_DIR=$DDK_PATH/toolkit/toolchain/hcc

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log_step() { echo -e "\n${GREEN}[STEP]${NC} $1"; }
log_ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
log_fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }

# ============================================================
# 1. 环境检查
# ============================================================
log_step "1/5: 检查环境依赖"

PYTHON=""
for candidate in /root/miniconda3/envs/operator-dev/bin/python3 /usr/bin/python3; do
    if $candidate -c "import numpy; from Crypto.Cipher import AES" 2>/dev/null; then
        PYTHON=$candidate; break
    fi
done
[ -z "$PYTHON" ] && log_fail "找不到同时有 numpy 和 pycryptodome 的 Python"
log_ok "Python: $PYTHON ($($PYTHON --version 2>&1))"

command -v atc &>/dev/null || log_fail "atc 不在 PATH 中"
log_ok "atc: $(which atc)"

[ -f "$DDK_PATH/runtime/include/acl/acl.h" ] || log_fail "找不到 AscendCL 头文件"
log_ok "AscendCL 头文件: $DDK_PATH/runtime/include/acl/acl.h"

[ -d "$TOOLCHAIN_DIR/bin" ] || log_fail "找不到 cross-compiler toolchain"
log_ok "Toolchain: $TOOLCHAIN_DIR"

# ============================================================
# 2. 编译算子
# ============================================================
log_step "2/5: 编译 AICPU 算子"

export TOOLCHAIN_DIR=$TOOLCHAIN_DIR
export ASCEND_OPP_PATH=$DDK_PATH/opp
export ASCEND_AICPU_PATH=$DDK_PATH
export ASCEND_TENSOR_COMPILER_INCLUDE=$DDK_PATH/compiler/include
export AICPU_KERNEL_TARGET=aes_ctr_decrypt_aicpu
export OPP_CUSTOM_VENDOR=customize_aicpu
export SYSTEM_INFO=openeuler_aarch64
export AICPU_SOC_VERSION=$SOC_VERSION

# rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
# cd "$BUILD_DIR"
# cmake .. \
#   -DCMAKE_CXX_FLAGS="-nostdinc++ \
#     -isystem ${TOOLCHAIN_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0 \
#     -isystem ${TOOLCHAIN_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu \
#     -D_GLIBCXX_USE_CXX11_ABI=1" 2>&1

# make -j$(nproc) 2>&1

cd "$AICPU_DIR"
bash build.sh
cd "$BUILD_DIR"

SO_FILE=$(find "$BUILD_DIR/makepkg" -name "libaes_ctr_decrypt_aicpu.so" 2>/dev/null)
[ -n "$SO_FILE" ] || log_fail "编译未生成 libaes_ctr_decrypt_aicpu.so"
log_ok "编译成功: $SO_FILE"

# ============================================================
# 3. 安装算子
# ============================================================
log_step "3/5: 安装算子包"

RUN_FILE=$(ls "$BUILD_DIR"/*.run 2>/dev/null | head -1)
[ -n "$RUN_FILE" ] || log_fail "找不到 .run 安装包"

bash "$RUN_FILE" --quiet 2>&1
log_ok "安装成功: $(basename "$RUN_FILE")"

INSTALLED_SO=$DDK_PATH/opp/vendors/$OPP_CUSTOM_VENDOR/op_impl/cpu/aicpu_kernel/impl/libaes_ctr_decrypt_aicpu.so
[ -f "$INSTALLED_SO" ] || log_fail "安装后找不到算子 SO"
log_ok "算子 SO: $INSTALLED_SO"

# ============================================================
# 4. 编译测试程序 & ATC 转换 & 执行验证
# ============================================================
log_step "4/5: 编译测试程序并执行验证"

cd "$TEST_DIR"

rm -rf op_models input output fusion_result.json
mkdir -p op_models input output

$PYTHON scripts/gen_data.py 1024

export LC_ALL=C
export PATH=/root/miniconda3/envs/operator-dev/bin:$DDK_PATH/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/operator-dev/lib:$DDK_PATH/lib64
export PYTHONPATH=$DDK_PATH/python/site-packages
export ASCEND_OPP_PATH=$DDK_PATH/opp
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/$OPP_CUSTOM_VENDOR

atc --singleop="$TEST_DIR/aes_ctr_decrypt_aicpu_op.json" \
    --soc_version=$SOC_VERSION \
    --output="$TEST_DIR/op_models" 2>&1

OM_FILE=$(ls "$TEST_DIR/op_models"/*.om 2>/dev/null)
[ -n "$OM_FILE" ] && log_ok "ATC 转换成功: $(basename "$OM_FILE")" || log_fail "ATC 未生成 OM 文件"

NPU_HOST_LIB=$DDK_PATH/aarch64-linux/devlib
g++ -std=c++11 "$TEST_DIR/test_aclop_v2.cpp" \
    -I$DDK_PATH/runtime/include \
    -L$NPU_HOST_LIB \
    -o "$TEST_DIR/execute_aes_ctr_decrypt" \
    -lascendcl 2>&1

# 算子调用工程编译
ld_path=$DDK_PATH/runtime/lib64:$DDK_PATH/lib64
LD_LIBRARY_PATH=$ld_path \
ASCEND_OPP_PATH=$DDK_PATH/opp \
"$TEST_DIR/execute_aes_ctr_decrypt" 2>&1

# ============================================================
# 5. 验证结果
# ============================================================
log_step "5/5: 验证结果"

$PYTHON "$TEST_DIR/scripts/verify_result.py" \
    "$TEST_DIR/output/result.bin" \
    "$TEST_DIR/output/golden.bin"

if [ $? -eq 0 ]; then
    log_ok "验证通过"
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  编译、安装、验证全部通过！${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    log_fail "验证失败"
fi
