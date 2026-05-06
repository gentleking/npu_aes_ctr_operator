#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$CURRENT_DIR/.." && pwd)

# ========== 配置区 ==========
# ATC 相关
SOC_VERSION=Ascend310P3
OP_JSON=$CURRENT_DIR/aes_ctr_decrypt_aicpu_op.json
OM_OUT_DIR=$CURRENT_DIR/op_models

# 测试数据
DATA_SIZE=1024
CIPHER_FILE=$CURRENT_DIR/input/ciphertext.bin
KEY_FILE=$CURRENT_DIR/input/key.bin
GOLDEN_FILE=$CURRENT_DIR/output/golden.bin
RESULT_FILE=$CURRENT_DIR/output/result.bin

# 程序
EXECUTABLE=$CURRENT_DIR/execute_aes_ctr_decrypt

# 日志颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_step()  { echo -e "\n${GREEN}[STEP]${NC} $1"; }
log_ok()    { echo -e "  ${GREEN}✓${NC} $1"; }
log_fail()  { echo -e "  ${RED}✗${NC} $1"; }

# ---------- 1. 环境检查 ----------
log_step "1/6: 检查环境依赖"

# 找 Python
PYTHON=""
for candidate in /root/miniconda3/envs/operator-dev/bin/python3 /usr/local/python3.7.5/bin/python3 /usr/bin/python3; do
    if $candidate -c "import numpy; import scipy" 2>/dev/null; then
        PYTHON=$candidate
        break
    fi
done
if [ -z "$PYTHON" ]; then
    log_fail "找不到同时有 numpy 和 scipy 的 Python"
    exit 1
fi
log_ok "Python: $PYTHON ($($PYTHON --version 2>&1))"

# 检查 atc
if ! command -v atc &>/dev/null; then
    log_fail "atc 不在 PATH 中"
    exit 1
fi
log_ok "atc: $(which atc)"

# 检查 AscendCL 头文件和库
DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
NPU_HOST_LIB=$DDK_PATH/aarch64-linux/devlib
if [ ! -f "$DDK_PATH/runtime/include/acl/acl.h" ]; then
    log_fail "找不到 AscendCL 头文件"
    exit 1
fi
log_ok "AscendCL 头文件: $DDK_PATH/runtime/include/acl/acl.h"

# ---------- 2. 生成测试数据 ----------
log_step "2/6: 生成测试数据"
mkdir -p $CURRENT_DIR/input $CURRENT_DIR/output
$PYTHON $CURRENT_DIR/scripts/gen_data.py $DATA_SIZE
if [ ! -f "$CIPHER_FILE" ] || [ ! -f "$KEY_FILE" ] || [ ! -f "$GOLDEN_FILE" ]; then
    log_fail "测试数据生成失败"
    exit 1
fi
log_ok "测试数据已生成 ($DATA_SIZE 字节)"

# ---------- 3. ATC 转换 ----------
log_step "3/6: ATC 转换单算子离线模型"
export LC_ALL=C
export PATH=/root/miniconda3/envs/operator-dev/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/operator-dev/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages
export ASCEND_OPP_PATH=$DDK_PATH/opp
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/customize_aicpu

rm -rf "$OM_OUT_DIR" "$CURRENT_DIR/fusion_result.json"
atc --singleop="$OP_JSON" --soc_version=$SOC_VERSION --output="$OM_OUT_DIR" 2>&1
if [ $? -ne 0 ]; then
    log_fail "ATC 转换失败"
    exit 1
fi
OM_FILE=$(ls "$OM_OUT_DIR"/*.om 2>/dev/null)
if [ -z "$OM_FILE" ]; then
    log_fail "ATC 未生成 OM 文件"
    exit 1
fi
log_ok "ATC 转换成功: $(basename "$OM_FILE")"

# ---------- 4. 编译调用程序 ----------
log_step "4/6: 编译验证程序"
g++ -std=c++11 "$CURRENT_DIR/test_aclop_v2.cpp" \
    -I$DDK_PATH/runtime/include \
    -L$NPU_HOST_LIB \
    -o "$EXECUTABLE" \
    -lascendcl \
    2>&1
if [ $? -ne 0 ]; then
    log_fail "编译失败"
    exit 1
fi
log_ok "编译成功: $EXECUTABLE"

# ---------- 5. 执行算子 ----------
log_step "5/6: 执行算子验证"
ld_path=/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64
ld_path=$ld_path:/usr/local/Ascend/ascend-toolkit/latest/lib64

LD_LIBRARY_PATH=$ld_path \
ASCEND_OPP_PATH=$DDK_PATH/opp \
$EXECUTABLE 2>&1
if [ $? -ne 0 ]; then
    log_fail "算子执行失败"
    exit 1
fi
log_ok "算子执行完成"

# ---------- 6. 验证结果 ----------
log_step "6/6: 验证结果"
$PYTHON "$CURRENT_DIR/scripts/verify_result.py" "$RESULT_FILE" "$GOLDEN_FILE"
if [ $? -ne 0 ]; then
    log_fail "验证失败"
    exit 1
fi
log_ok "验证通过"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  全部验证通过！${NC}"
echo -e "${GREEN}========================================${NC}"
