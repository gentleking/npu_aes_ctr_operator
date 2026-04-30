"""
AesCtrDecryptAicpu 算子测试数据生成脚本

本脚本使用 PyCryptodome 库生成 AES-CTR 加密的测试数据。
通过与标准 AES 实现的对比，可以验证自定义算子的正确性。

生成的文件:
  - input/ciphertext.bin:  待解密的密文数据（算子输入 0）
  - input/key.bin:         AES-128 密钥（算子输入 1）
  - output/golden.bin:     解密期望结果（用于与算子输出对比）

用法:
  python3 gen_data.py [data_len]

参数:
  data_len: 生成数据的字节数（默认 1024）

注意事项:
  - np.arange 必须指定 dtype=np.uint8，否则默认 int64
  - 不要对 uint8 数组做 % 256 运算，会导致 NumPy 提升为 int16
  - 文件路径相对于 AclNNInvocation 目录
"""
import numpy as np
import sys
from Crypto.Cipher import AES


def gen_data(data_len):
    """
    生成 AES-CTR 加解密测试数据

    流程:
      1. 生成从 0 递增的 uint8 序列作为原始明文
      2. 使用固定密钥和 IV=0 进行 AES-CTR 加密
      3. 保存密文（算子输入）、密钥（算子输入）、明文（golden 对比值）
    """

    # ---- 1. 固定密钥（16 字节） ----
    # 使用与算子实现完全相同的密钥
    key = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                    0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10],
                   dtype=np.uint8)

    # ---- 2. 初始向量（16 字节全零） ----
    # 与算子实现中的 IV 保持一致
    iv = np.zeros(16, dtype=np.uint8)

    # ---- 3. 生成原始明文数据 ----
    # 【注意】必须指定 dtype=np.uint8
    # 不允许做 % 256，否则 NumPy 会隐式转换为 int16
    plaintext = np.arange(data_len, dtype=np.uint8)

    # ---- 4. 使用 PyCryptodome 进行 AES-CTR 加密 ----
    # AES.MODE_CTR 需要 initial_value 和 nonce 参数
    # initial_value 作为计数器的初始值
    # nonce 为空时计数器由 initial_value 开始
    cipher = AES.new(key.tobytes(), AES.MODE_CTR,
                     initial_value=iv.tobytes(), nonce=b'')

    # encrypt 输出与输入等长的密文（CTR 模式不做 padding）
    ciphertext = cipher.encrypt(plaintext.tobytes())

    # ---- 5. 保存文件 ----
    # golden: 原始明文（算子应输出此值）
    plaintext.tofile("./output/golden.bin")

    # key: 加密/解密密钥
    key.tofile("./input/key.bin")

    # ciphertext: 密文（算子输入）
    with open("./input/ciphertext.bin", "wb") as f:
        f.write(ciphertext)

    print(f"Generated {data_len} bytes: "
          f"input/ciphertext.bin, input/key.bin, output/golden.bin")


if __name__ == "__main__":
    data_len = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    gen_data(data_len)
