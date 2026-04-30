"""
AesCtrDecryptAicpu 算子执行结果验证脚本

比较算子输出文件（result.bin）与 golden 文件（golden.bin），
验证解密结果的正确性。

用法:
  python3 verify_result.py <output_file> <golden_file>

返回值:
  0: 验证通过（结果完全一致）
  1: 验证失败（存在差异）

注意:
  - 两个文件的数据类型必须都是 uint8
  - 文件大小不一致时直接判定失败
"""
import sys
import numpy as np


def verify_result(output_file, golden_file):
    """验证算子输出与 golden 值是否一致"""

    # 读取算子输出和 golden 文件
    out_data = np.fromfile(output_file, dtype=np.uint8)
    gd_data = np.fromfile(golden_file, dtype=np.uint8)

    # 检查尺寸是否一致
    if out_data.shape != gd_data.shape:
        print(f"Size mismatch: output={out_data.shape}, "
              f"golden={gd_data.shape}")
        return False

    # 逐字节比较
    diff = (out_data != gd_data).sum()
    if diff > 0:
        print(f"FAIL: {diff} bytes differ")
        return False

    print("PASS: output matches golden!")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)
    res = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if res else 1)
