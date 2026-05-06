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
import hashlib


def file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_result(output_file, golden_file):
    out_hash = file_hash(output_file)
    gd_hash = file_hash(golden_file)

    print(f"  SHA256(output): {out_hash}")
    print(f"  SHA256(golden): {gd_hash}")

    if out_hash == gd_hash:
        print("PASS: output matches golden!")
        return True
    else:
        print("FAIL: output differs from golden!")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)
    res = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if res else 1)
