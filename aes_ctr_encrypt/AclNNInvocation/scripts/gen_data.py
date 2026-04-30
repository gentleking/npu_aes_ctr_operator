import numpy as np
from Crypto.Cipher import AES

def convert_size(size_bytes):
    """将字节数转换为人类可读的单位"""
    if size_bytes == 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(units)-1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {units[i]}"

def gen_data():
    # 目标数据量: 修改这个值即可调整数据规模
    data_len_half = 1024 * 8

    # 生成固定明文(使用递增值而非随机值)
    plaintext = np.arange(data_len_half, dtype=np.uint16) % 65535
    
    # 固定的 key (16字节)
    key = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                    0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10], dtype=np.uint8)
    
    # 固定的 iv (16字节)
    # iv = np.array([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 
    #                0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF], dtype=np.uint8)
    iv = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8)

    # 用 PyCryptodome 生成 golden 数据
    cipher = AES.new(key.tobytes(), AES.MODE_CTR, initial_value=iv.tobytes(), nonce=b'')
    golden = cipher.encrypt(plaintext.tobytes())  # bytes

    # 保存生成的数据到文件
    plaintext.tofile("./input/plaintext.bin")
    key.tofile("./input/key.bin")
    # iv.tofile("./input/iv.bin")
    with open("./output/golden.bin", "wb") as f:
        f.write(golden)
    
    # 动态计算数据大小并输出
    total_bytes = data_len_half * plaintext.itemsize  # 自动计算总字节数
    size_str = convert_size(total_bytes)
    print(f"Generated {size_str} data: plaintext.bin, key.bin, golden.bin")

if __name__ == "__main__":
    gen_data()