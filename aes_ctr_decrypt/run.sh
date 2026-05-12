#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/root/miniconda3/envs/operator-dev/bin/python3"

echo "============================================"
echo " AES-CTR Decrypt 算子 — 完整流程"
echo "============================================"
echo ""

# ---- 1. 编译算子 ----
echo "[1/5] 编译算子 ..."
cd "$SCRIPT_DIR/AesCtrDecrypt"
rm -rf build_out
bash build.sh > /tmp/aes_build.log 2>&1
echo "      编译完成"

# ---- 2. 安装算子 ----
echo "[2/5] 安装算子 ..."
cd build_out
bash custom_opp_openEuler_aarch64.run > /tmp/aes_install.log 2>&1
echo "      安装完成"

# ---- 3. 设置环境变量 ----
echo "[3/5] 设置环境变量 ..."
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_aes_ctr/op_api/lib/:${LD_LIBRARY_PATH}
echo "      LD_LIBRARY_PATH 已更新"

# ---- 4. Kernel正确性验证 ----
echo "[4/5] Kernel正确性验证 ..."
cd "$SCRIPT_DIR"
$PYTHON -c "
import torch, torch_npu
from Crypto.Cipher import AES
import hashlib

key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
iv = torch.zeros((1, 16), dtype=torch.uint8)
passed = 0
for size in [16384, 32768, 65536, 1048576]:
    p = torch.randint(0, 256, size=(size,), dtype=torch.uint8)
    c = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
    g = torch.frombuffer(bytearray(c.encrypt(p.numpy().tobytes())), dtype=torch.uint8)
    o = torch_npu.npu_aes_ctr_decrypt(p.npu(), key.npu()).cpu()
    m = hashlib.md5(o.numpy().tobytes()).hexdigest() == hashlib.md5(g.numpy().tobytes()).hexdigest()
    print(f'      size={size:>7}B: {\"PASS\" if m else \"FAIL\"}')
    passed += m
print(f'      结果: {passed}/4 PASS')
if passed != 4:
    print('      错误: Kernel验证失败')
    exit(1)
"
echo "      验证通过"

# ---- 5. 运行基准测试 ----
echo "[5/5] 运行基准测试 ..."
$PYTHON "$SCRIPT_DIR/benchmark_all.py"

echo ""
echo "============================================"
echo " 全部完成"
echo "============================================"
