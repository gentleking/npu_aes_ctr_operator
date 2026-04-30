from Crypto.Cipher import AES
import hashlib
import numpy as np
import torch
import torch_npu


class AESCipher:
    """AES加密解密工具类"""
    
    @staticmethod
    def encrypt_tensor_ref(plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """模拟AES加密（这里用XOR简化，实际应该用真正的AES）"""

        # 生成全0的 iv，形状 [1, 16]，类型 uint8
        iv = torch.zeros((1, 16), dtype=torch.uint8)

        cipher = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
        ciphertext = cipher.encrypt(plaintext.numpy().tobytes())
        np_array = np.frombuffer(bytearray(ciphertext), dtype=np.uint8)
        cipher_tensor = torch.from_numpy(np_array).reshape(plaintext.size())
        
        print("cipher tensor: ", cipher_tensor, '\n', "cypher tensor size: ", cipher_tensor.size())
        return cipher_tensor
    
    @staticmethod
    def decrypt_tensor_custom(cipher_tensor: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """使用自定义算子进行AES解密"""
        if torch.npu.is_available():
            try:
                # 使用你的自定义解密算子
                return torch_npu.npu_aes_ctr_decrypt(
                    cipher_tensor.npu(), 
                    key.npu()
                ).cpu()
            except Exception as e:
                print(f"NPU解密失败: {e}")
                return cipher_tensor
        else:
            # NPU不可用，使用CPU模拟
            # return cipher_tensor ^ key[:cipher_tensor.numel()].reshape(cipher_tensor.shape)
            # raise RuntimeError("NPU不可用，无法进行AES解密")
            print("NPU不可用")
            return cipher_tensor
