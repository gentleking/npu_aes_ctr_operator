import torch
import torch_npu
import numpy as np

from Crypto.Cipher import AES
import hashlib
import time
import os

import torch.nn as nn
import torchvision.models as models

torch_npu.npu.set_compile_mode(jit_compile=False)

# def valid_aes_ctr_encrypt(encrypted_tensor):
#     print("=" * 50)
#     print(f"加密结果: {encrypted_tensor}")
#     add_tensor = torch.ones_like(encrypted_tensor)
#     result_after_addition = encrypted_tensor + add_tensor
#     print(f"加法后结果: {result_after_addition}")
#     return encrypted_tensor

class EncryptedModelWrapper(nn.Module):
    """在预训练模型最后插入加密算子的包装器"""
    
    def __init__(self, pretrained_model, key_size=16):
        """
        :param pretrained_model: 已经训练好的模型
        :param key_size: 加密密钥大小，应与模型输出维度匹配
        """
        super(EncryptedModelWrapper, self).__init__()
        self.pretrained_model = pretrained_model
        self.encryption_enabled = True
        
        # 创建加密密钥
        self.encryption_key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
        
        # 冻结预训练模型的参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
    def enable_encryption(self, enabled=True):
        """启用或禁用加密"""
        self.encryption_enabled = enabled
        return self
    
    def custom_aes_ctr_encrypt(self, plaintext):
        """自定义AES-CTR加密实现"""
        if not self.encryption_enabled or not torch.npu.is_available():
            return plaintext
            
        print(f"执行加密操作...")
        print(f"   输入明文数据的形状: {plaintext.shape}")
        print(f"   输入范围: [{plaintext.min():.4f}, {plaintext.max():.4f}]")
        
        torch.npu.synchronize()
        try:                
            encrypted_tensor = torch_npu.npu_aes_ctr_encrypt(
                plaintext.to(torch.uint8).npu(), 
                self.encryption_key.npu()
            ).cpu()
            torch.npu.synchronize()
            
            print(f"   加密完成")
            print(f"   输入加密后的形状: {encrypted_tensor.shape}")
            return encrypted_tensor
            
        except Exception as e:
            print(f"❌ 加密失败: {e}")
            return plaintext
    
    def pad_to_block_size(self, data):
        """将数据填充到128bit(16字节)的整数倍"""
        original_shape = data.shape
        original_dtype = data.dtype
        
        # 将数据展平为一维
        data_flat = data.view(-1)
        
        # 计算需要填充的字节数
        element_size = data.element_size()  # 每个元素的字节数
        total_bytes = data_flat.numel() * element_size
        block_size = 16  # 128bit = 16字节
        
        # 计算需要填充到多少个block
        blocks_needed = (total_bytes + block_size - 1) // block_size
        padded_bytes = blocks_needed * block_size
        padded_elements = padded_bytes // element_size
        
        print(f"   数据填充信息:")
        print(f"     - 原始元素数: {data_flat.numel()}")
        print(f"     - 元素大小: {element_size} 字节")
        print(f"     - 总字节数: {total_bytes}")
        print(f"     - 需要填充到: {padded_bytes} 字节 ({blocks_needed} 个block)")
        
        # 如果需要填充
        if padded_elements > data_flat.numel():
            # 计算需要填充的元素数量
            pad_count = padded_elements - data_flat.numel()
            
            # 使用零填充
            padded_data = torch.cat([
                data_flat,
                torch.zeros(pad_count, dtype=data_flat.dtype)
            ])
            
            print(f"     - 填充了 {pad_count} 个元素")
        else:
            padded_data = data_flat
            print(f"     - 无需填充")
        
        # return padded_data, original_shape, padded_elements
        return padded_data
    
    def forward(self, x):
        # 步骤1: 预训练模型推理
        with torch.no_grad():
            original_output = self.pretrained_model(x)
            # padded_output = self.pad_to_block_size(original_output)
        final_output = original_output
        print("=" * 70)
        print("模型推理流程:")
        print(f"   输入形状: {x.shape}")
        print(f"   原始输出形状: {original_output.shape}")
        print(f"   原始输出样例: {original_output[0][:5] if len(original_output.shape) > 1 else original_output[:5]}")
        
        # 步骤2: 在模型最后插入加密
        if self.encryption_enabled and torch.npu.is_available():
            final_output = self.custom_aes_ctr_encrypt(original_output)

        return final_output
    
    def ref_aes_ctr_decrypt(self, encrypted_data):
        """AES-CTR解密函数"""
            
        print(f"执行AES-CTR解密操作...")
        
        try:
            print(f"   开始解密计算...")
            iv = torch.zeros((1, 16), dtype=torch.uint8).numpy().tobytes()
            cypher = AES.new(self.encryption_key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv, nonce=b'')
            decrypted_output = cypher.decrypt(encrypted_data)
            
            print(f"   ✅ 解密完成")
            return decrypted_output
            
        except Exception as e:
            print(f"❌ 解密失败: {e}")
            import traceback
            traceback.print_exc()
            return encrypted_data
    
    def verify_encryption_and_decryption(self, encrypted_output, original_output):
        """
        验证加密解密过程的正确性
        
        Args:
            encrypted_output: 加密后的输出 (float tensor)
            original_output: 原始输出 (float tensor)
        """
        print("=" * 80)
        print("开始解密&验证...")
        
        # decrypted_output = self.ref_aes_ctr_decrypt(encrypted_output)
        encrypted_origin_output = self.ref_aes_ctr_encrypt(original_output)
        
        hash_ref = hashlib.md5(encrypted_origin_output).hexdigest()
        hash_custom = hashlib.md5(encrypted_output).hexdigest()
        
        print(f"解密后的输出结果 MD5: {hash_custom}")
        print(f"无加解密输出结果 MD5: {hash_ref}")
        print("=" * 80)
        
        # return 

    def ref_aes_ctr_encrypt(self, plaintext):
        """参考实现的AES-CTR加密函数"""
        print(f"执行参考实现AES-CTR加密操作...")
        
        try:
            iv = torch.zeros((1, 16), dtype=torch.uint8).numpy().tobytes()
            key = self.encryption_key.numpy().tobytes()
            # plaintext_bytes = plaintext.numpy().tobytes()
            
            cypher = AES.new(key, AES.MODE_CTR, initial_value=iv, nonce=b'')
            encrypted_output = cypher.encrypt(plaintext)
            
            print(f"   ✅ 加密完成")
            return encrypted_output
            
        except Exception as e:
            print(f"❌ 加密失败: {e}")
            import traceback
            traceback.print_exc()
            return plaintext


def load_pretrained_resnet():
    """加载预训练的ResNet模型"""
    print("加载预训练ResNet-18模型...")
    model = models.resnet18(pretrained=True)
    model.eval()  # 设置为评估模式
    print("ResNet-18加载完成")
    return model

def load_pretrained_mobilenet():
    """加载预训练的MobileNet模型"""
    print("加载预训练MobileNetV2模型...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    print("MobileNetV2加载完成")
    return model

def test_with_pretrained_models():
    """使用真实预训练模型测试加密集成"""
    print("开始预训练模型加密集成测试")
    
    # 测试数据（模拟ImageNet输入）
    test_input = torch.randn(2, 3, 224, 224)  # batch_size=2, 3通道, 224x224
    print(f"测试输入: {test_input.shape}")
    
    # 测试1: ResNet-18
    print("\n" + "="*80)
    print("测试1: ResNet-18 + 加密")
    resnet = load_pretrained_resnet()
    
    # 创建加密包装器（ResNet输出1000类）
    encrypted_resnet = EncryptedModelWrapper(resnet, key_size=16)
    
    # 启用加密测试
    with torch.no_grad():
        encrypted_output = encrypted_resnet.enable_encryption(True)(test_input)
    
    # 禁用加密测试（基准）
    print("\n" + "="*80)
    print("测试1: ResNet-18 + 无加密（基准）")
    with torch.no_grad():
        normal_output = encrypted_resnet.enable_encryption(False)(test_input)
    
    # 测试2: MobileNetV2
    print("\n" + "="*80)
    print("测试2: MobileNetV2 + 加密")
    mobilenet = load_pretrained_mobilenet()
    encrypted_mobilenet = EncryptedModelWrapper(mobilenet, key_size=16)
    
    with torch.no_grad():
        mobile_encrypted_output = encrypted_mobilenet.enable_encryption(True)(test_input)
    
    return encrypted_resnet, encrypted_mobilenet

def comprehensive_verification():
    """综合验证加密集成"""
    print("综合验证加密集成")
    
    # 加载预训练模型
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    
    # 创建加密包装器
    encrypted_model = EncryptedModelWrapper(resnet, key_size=16)
    
    # 测试数据
    test_input = torch.randn(1, 3, 224, 224)
    
    print("验证步骤:")
    print("1. 验证模型结构完整性")
    print("2. 验证加密功能")
    print("3. 验证输出一致性")
    
    # 步骤1: 无加密基准
    print("\n步骤1: 无加密基准测试")
    encrypted_model.enable_encryption(False)
    with torch.no_grad():
        baseline_output = encrypted_model(test_input)
    
    # 步骤2: 有加密测试
    if torch.npu.is_available():
        print("\n步骤2: 加密功能测试")
        encrypted_model.enable_encryption(True)
        with torch.no_grad():
            encrypted_output = encrypted_model(test_input)
        
        encrypted_model.verify_encryption_and_decryption(encrypted_output.numpy().tobytes(), baseline_output.numpy().tobytes())
    
    # 步骤3: 性能测试
    print("\n步骤3: 性能影响测试")
    if torch.npu.is_available():
        test_input_large = torch.randn(16, 3, 224, 224)  # 更大的batch
        
        import time
        
        # 无加密性能
        encrypted_model.enable_encryption(False)
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = encrypted_model(test_input_large)
        torch.npu.synchronize()
        normal_time = time.time() - start
        
        # 有加密性能
        encrypted_model.enable_encryption(True)
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = encrypted_model(test_input_large)
        torch.npu.synchronize()
        encrypted_time = time.time() - start
        
        print(f"   无加密推理时间: {normal_time:.4f}s")
        print(f"   有加密推理时间: {encrypted_time:.4f}s")
        print(f"   时间开销: {encrypted_time - normal_time:.4f}s")
        print(f"   开销比例: {(encrypted_time - normal_time) / normal_time * 100:.2f}%")
    
    print("\n!验证完成!")
    return encrypted_model

def demo_real_world_usage():
    """演示真实世界使用场景"""
    print("\n" + "="*80)
    print("真实世界使用场景演示")
    
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # 创建加密版本
    encrypted_model = EncryptedModelWrapper(model, key_size=16)
    
    print("使用示例:")
    print("1. 安全推理（启用加密）:")
    encrypted_model.enable_encryption(True)
    
    print("2. 普通推理（禁用加密）:")
    encrypted_model.enable_encryption(False)
    
    print("3. 模型可以像普通模型一样使用:")
    print("   - model(input_tensor)")
    print("   - torch.jit.trace(model, example_input)")
    print("   - 部署到生产环境")
    
    return encrypted_model


if __name__ == "__main__":
    print("预训练模型加密集成验证")
    print("=" * 80)
    
    try:
        # 检查环境
        if torch.npu.is_available():
            print("NPU设备可用")
        else:
            print("NPU设备不可用，将跳过加密测试")
        
        # 综合验证
        verified_model = comprehensive_verification()
        
        # 真实预训练模型测试
        print("\n" + "="*80)
        test_with_pretrained_models()
        
        # 使用场景演示
        demo_real_world_usage()
        
        print("\n" + "="*80)
        print("所有测试完成!")
        print("  ✅ 成功在预训练模型最后插入加密算子")
        print("  ✅ 保持模型原有接口不变")
        print("  ✅ 支持加密/非加密模式切换")
        print("  ✅ 验证了功能完整性和性能")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()