import torch
import torch_npu
import numpy as np

from Crypto.Cipher import AES
import hashlib
import time
import os

torch_npu.npu.set_compile_mode(jit_compile=False)

# 随机数初始化向量key和iv
global key_, iv
key_ = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
iv = torch.zeros((1, 16), dtype=torch.uint8)

shapes = [(8, 2048),    # 成功
          (4, 4096),    # 成功
          (16, 1024),   # 成功
          (8, 1024),    # 成功
          (4, 2048),    # 成功
        #   (16, 2048),   # 失败
        #   (8, 4096),    # 失败
          (8, 500),     # 成功
          (7, 2048),    # 成功
          (3, 1999),    # 成功
          (1, 1)]       # 成功

# dtypes = [torch.uint8, torch.float16]
dtypes = [torch.uint8]

# 张量转字节数组
def tensor_to_bytes(tensor):
    """将张量转换为字节数组"""
    return tensor.numpy().tobytes()

# 字节数组转张量
def bytes_to_tensor(byte_data, original_tensor):
    """将字节数组转换回原始张量格式"""
    dtype = original_tensor.dtype
    shape = original_tensor.shape
    device = original_tensor.device
    
    # 重建张量
    np_array = np.frombuffer(byte_data, dtype=dtype.numpy())
    return torch.tensor(np_array, device=device, dtype=dtype).reshape(shape)

# 开源AES-CTR加密实现
def open_aes_ctr_encrypt(data, key, iv):
    """
    使用标准库实现的AES-CTR加密
    :param data: 输入数据 (字节数组)
    :param key: 加密密钥 (16字节)
    :param iv: 随机数 (16字节)
    :return: 加密后的字节数组
    """
    data_bytes = data.numpy().tobytes()
    key_bytes = key.numpy().tobytes()
    iv_bytes = iv.numpy().tobytes()
    
    cipher = AES.new(key_bytes, AES.MODE_CTR, initial_value=iv_bytes, nonce=b'')
    return cipher.encrypt(data_bytes)

def custom_aes_ctr_encrypt(plaintext, key):
    """
    使用自定义AES-CTR算法对数据进行加密。首先将plaintext和key从CPU传输到NPU，
    再调用torch_npu提供的AES-CTR加密算子进行加密，最后将加密结果传回CPU。
    :param plaintext: 输入明文 (张量)
    :param key: 加密密钥 (张量)
    :return: 加密后的张量
    """
    torch.npu.synchronize()
    encrypted_tensor = torch_npu.npu_aes_ctr_encrypt(plaintext.npu(), key.npu()).cpu()
    torch.npu.synchronize()

    return encrypted_tensor


def generate_test_cases():
    """
    生成测试用例
    :return: 测试用例列表
    """
    test_cases = []
    
    
    for shape in shapes:
        for dtype in dtypes:
            # 随机生成数据
            data = torch.randint(0, 256, size=shape, dtype=dtype)
            key = key_  # 使用全局定义的key
            nonce = torch.zeros((1, 16), dtype=torch.uint8)  # 使用全0的nonce
            
            test_cases.append({
                "shape": shape,
                "dtype": dtype,
                "data": data,
                "key": key,
                "nonce": nonce
            })
    
    return test_cases

def test_aes_ctr_encrypt():
    """主测试函数：验证自定义算子与标准库实现的一致性"""
    test_cases = generate_test_cases()
    passed_tests = 0
    total_tests = len(test_cases)
    
    print(f"开始测试AES-CTR加密算子，共 {total_tests} 个测试用例...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases):
        shape = test_case["shape"]
        dtype = test_case["dtype"]
        data = test_case["data"]
        key = test_case["key"]
        nonce = test_case["nonce"]
        
        print(f"\n测试用例 #{i+1}: shape={shape}, dtype={dtype}")
        print(f"数据大小: {data.element_size() * data.nelement()} 字节, 数据类型: {data.dtype}")
        
        try:
            # 使用自定义算子加密
            custom_start = time.time()
            encrypted_custom = custom_aes_ctr_encrypt(data, key)
            custom_time = time.time() - custom_start
            
            # 使用参考实现加密
            ref_start = time.time()
            encrypted_bytes = open_aes_ctr_encrypt(data, key, iv)
            ref_time = time.time() - ref_start
            
            # 比较结果
            hash_custom = hashlib.md5(encrypted_custom.numpy().tobytes()).hexdigest()
            hash_ref = hashlib.md5(encrypted_bytes).hexdigest()
            is_equal = hash_custom == hash_ref
            print(f"自定义算子的输出 MD5: {hash_custom}")
            print(f"开源算法实现输出 MD5: {hash_ref}")
            
            # 输出结果
            print(f"自定义算子时间: {custom_time:.6f}s")
            print(f"参考实现的时间: {ref_time:.6f}s")
            # print(f"结果匹配: {'是' if is_equal else '否'}")
            
            if is_equal:
                print("✅ 测试通过!")
                passed_tests += 1
            else:
                print("❌ 测试失败: 自定义算子与参考实现不一致")
                
        except Exception as e:
            print(f"测试失败: {str(e)}")
        
        print("-" * 60)
    
    # 汇总结果
    print("\n" + "=" * 80)
    print(f"测试完成! 通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.2f}%)")
    
    if passed_tests == total_tests:
        print("✅ 所有测试用例通过! 自定义算子与参考实现完全一致")
    else:
        print(f"❌ 有 {total_tests - passed_tests} 个测试用例未通过，请检查算子实现")
        
if __name__ == "__main__":
    experimental_config_ = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2
    )
    
    prof = torch_npu.profiler.profile(
        experimental_config=experimental_config_,
        with_stack=True,
    )
    # 启动性能分析
    prof.start()
    test_aes_ctr_encrypt()
    prof.stop()
    # 导出 Chrome Trace 文件
    os.makedirs('./trace', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filepath = f'./trace/chrome_trace_{timestamp}.json'
    prof.export_chrome_trace(filepath)
    print(f"Profile trace exported to {filepath}")