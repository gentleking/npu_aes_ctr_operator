import torch
import torch.nn as nn
import torch_npu
import torchvision.models as models
import numpy as np
from typing import Tuple, Dict, Any
import time
# import utils
from Crypto.Cipher import AES

class AESCipher:
    """AES加密解密工具类"""
    
    @staticmethod
    def encrypt_tensor_ref(plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """CPU使用open AES-CTR算法对输出plaintext tensor进行加密
        
        """
        # 生成全0的 iv，形状 [1, 16]，类型 uint8
        iv = torch.zeros((1, 16), dtype=torch.uint8)

        cipher = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
        ciphertext = cipher.encrypt(plaintext.numpy().tobytes())
        
        # 将加密结果转换为tensor，形状与输入相同
        np_array = np.frombuffer(bytearray(ciphertext), dtype=np.uint8)
        cipher_tensor = torch.from_numpy(np_array).reshape(plaintext.size())
        
        print("cipher images tensor: ", cipher_tensor, '\n', "cypher images tensor size: ", cipher_tensor.size())
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
            print("NPU不可用")
            return cipher_tensor

class EncryptedResNetWrapper(nn.Module):
    """在ResNet输入前加入AES解密的包装器"""
    
    def __init__(self, base_model: nn.Module, key: torch.Tensor, model_name: str):
        super().__init__()
        self.base_model = base_model
        self.key = key
        self.model_name = model_name
        
        # 冻结基础模型参数（只推理，不训练）
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 设置为评估模式
        self.base_model.eval()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：先将输入的加密tensor进行解密，再通过原本模型进行推理
        
        Args:
            x: 加密的输入tensor
        
        Returns:
            模型输出
        """
        # 步骤1: 解密输入
        decryption_start = time.time()
        
        if torch.npu.is_available():
            # 使用NPU解密
            x_npu = x.npu()
            key_npu = self.key.npu()
            
            # 同步确保数据传输完成
            torch.npu.synchronize()
            
            # 执行NPU解密
            x_decrypted_npu = torch_npu.npu_aes_ctr_decrypt(x_npu, key_npu)
            
            # 转移回CPU
            x_decrypted = x_decrypted_npu.cpu()
            torch.npu.synchronize()
        else:
            # 
            print("NPU不可用")
        
        decryption_time = time.time() - decryption_start
        
        # 步骤2: 模型推理
        with torch.no_grad():
            model_start = time.time()

            # 将解密后的uint8类型tensor转变为float类型并归一化。序列化形状为(1, 3, 1, 512)
            x_decrypted_input = x_decrypted.view(torch.float32)
            x_decrypted_input = x_decrypted_input.view(1, 3, 1, 512)
            print("decrypted tensor float: ", x_decrypted_input, '\n', "decrypted tensor size: ", x_decrypted_input.size(), "\ndecrypted tensor dtype: ", x_decrypted_input.dtype)

            output = self.base_model(x_decrypted_input)
            model_time = time.time() - model_start
        
        # 记录性能信息
        if hasattr(self, '_decryption_times'):
            self._decryption_times.append(decryption_time)
            self._model_times.append(model_time)
        
        return output
    
    def reset_timers(self):
        """重置计时器"""
        self._decryption_times = []
        self._model_times = []
    
    def get_timing_stats(self) -> Dict[str, float]:
        """获取计时统计"""
        if not hasattr(self, '_decryption_times') or len(self._decryption_times) == 0:
            return {}
        
        return {
            'avg_decryption_ms': np.mean(self._decryption_times) * 1000,
            'avg_model_ms': np.mean(self._model_times) * 1000,
            'total_decryption_ms': np.sum(self._decryption_times) * 1000,
            'total_model_ms': np.sum(self._model_times) * 1000,
        }

def load_pretrained_model(model_type: str = 'resnet18') -> nn.Module:
    """加载预训练模型"""
    print(f"加载{model_type}模型...")
    
    try:
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model.eval()
        print(f"{model_type}加载完成")
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("使用模拟模型")
        # return create_mock_model()

def create_mock_model() -> nn.Module:
    """创建模拟模型（当无法下载预训练模型时使用）"""
    class MockResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 1000)
            
            # 初始化权重
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
            nn.init.normal_(self.fc.weight, 0, 0.01)
            nn.init.constant_(self.fc.bias, 0)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    model = MockResNet()
    model.eval()
    print("模拟模型创建完成")
    return model

def generate_test_data(batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成测试数据"""
    print(f"生成测试数据 (batch_size={batch_size})")
    
    # 生成随机图像数据（模拟ImageNet输入）
    plain_images = torch.randn(batch_size, 3, 1, 512)
    
    # 生成AES密钥
    key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
    print("plain images: ", plain_images, '\n', "plain images size: ", plain_images.size(), "\nplain images dtype: ", plain_images.dtype)
    print(f"   密钥形状: {key.shape}")
    
    return plain_images, key

def verify_encryption_decryption(
    model_wrapper: EncryptedResNetWrapper,
    plain_images: torch.Tensor,
    key: torch.Tensor
) -> bool:
    """
    验证加密解密流程的正确性
    
    Returns:
        bool: 验证是否通过
    """
    print("开始验证CPU加密，NPU解密流程...")
    
    try:
        # 步骤1: 加密原始图像
        print("  1. 加密原始图像...")
        plain_images_uint8 = plain_images.view(torch.uint8)
        cipher_images = AESCipher.encrypt_tensor_ref(
            plain_images_uint8, 
            key
        )
        
        # 步骤2: 使用包装器模型解密后推理
        print("  2. 使用加密模型推理...")
        wrapper_start = time.time()
        wrapper_output = model_wrapper(cipher_images)
        wrapper_time = time.time() - wrapper_start
        
        # 步骤3: 使用原始模型推理
        print("  3. 使用原始模型推理...")
        model_start = time.time()
        with torch.no_grad():
            original_output = model_wrapper.base_model(plain_images)
        model_time = time.time() - model_start
        
        # 步骤4: 比较结果
        print("  4. 比较结果...")
        
        # 计算相似度
        similarity = torch.cosine_similarity(
            wrapper_output.flatten(), 
            original_output.flatten(), 
            dim=0
        ).item()
        
        # 计算绝对误差
        abs_error = torch.abs(wrapper_output - original_output).max().item()
        
        # 判断是否通过
        threshold = 1e-4
        is_correct = abs_error < threshold
        
        print(f"验证结果:")
        print(f"   包装器推理时间: {wrapper_time*1000:.2f} ms")
        print(f"   原始模型推理时间: {model_time*1000:.2f} ms")
        print(f"   余弦相似度: {similarity:.6f}")
        print(f"   最大绝对误差: {abs_error:.6f}")
        print(f"   阈值: {threshold}")
        print(f"   验证结果: {'✅ 通过' if is_correct else '❌ 失败'}")
        
        if not is_correct:
            print(f"⚠️  误差较大，可能原因:")
            print(f"   - 解密算子实现可能有误")
            print(f"   - 数据类型转换问题")
            print(f"   - NPU与CPU计算差异")
        
        return is_correct
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_model(
    model_wrapper: EncryptedResNetWrapper,
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    对模型进行基准测试
    
    Returns:
        包含性能指标的字典
    """
    print(f"🏃 开始基准测试 ({num_iterations}次迭代)")
    
    # 生成测试数据
    batch_size = 4
    plain_images, key = generate_test_data(batch_size)
    cipher_images = AESCipher.encrypt_tensor_ref(plain_images.to(torch.uint8), key)
    
    # 重置计时器
    model_wrapper.reset_timers()
    
    # 预热
    print("  预热运行...")
    for _ in range(2):
        _ = model_wrapper(cipher_images.float())
    
    # 正式测试
    print("  正式测试...")
    test_start = time.time()
    
    for i in range(num_iterations):
        if i % max(1, num_iterations // 10) == 0:
            print(f"  进度: {i+1}/{num_iterations}")
        
        output = model_wrapper(cipher_images.float())
    
    total_time = time.time() - test_start
    
    # 获取统计信息
    stats = model_wrapper.get_timing_stats()
    
    print(f"📊 基准测试结果:")
    print(f"   总时间: {total_time*1000:.2f} ms")
    print(f"   平均每批时间: {total_time/num_iterations*1000:.2f} ms")
    print(f"   平均解密时间: {stats.get('avg_decryption_ms', 0):.2f} ms")
    print(f"   平均模型时间: {stats.get('avg_model_ms', 0):.2f} ms")
    print(f"   FPS: {batch_size * num_iterations / total_time:.2f}")
    
    return {
        'total_time_ms': total_time * 1000,
        'avg_batch_time_ms': total_time / num_iterations * 1000,
        'avg_decryption_ms': stats.get('avg_decryption_ms', 0),
        'avg_model_ms': stats.get('avg_model_ms', 0),
        'fps': batch_size * num_iterations / total_time,
    }

def main():
    """主函数"""
    print("AES解密算子验证ResNet模型推理")
    print("=" * 60)
    
    # 检查环境

    if torch.npu.is_available():
        print(f"   NPU可用: {torch.npu.is_available()}")
        print(f"   NPU设备数量: {torch.npu.device_count()}")
    
    # 生成测试数据
    plain_images, key = generate_test_data(batch_size=1)
    
    # 测试的模型列表
    model_types = ['resnet18', 'resnet50']
    
    all_results = {}
    
    for model_type in model_types:
        print(f"===========测试模型: {model_type}===========")        
        try:
            # 加载模型
            model = load_pretrained_model(model_type)
            
            # 创建加密包装器
            model_wrapper = EncryptedResNetWrapper(model, key, model_type)
            
            # 验证加密解密流程
            print(f"\n验证{model_type}...")
            verification_passed = verify_encryption_decryption(
                model_wrapper, 
                plain_images, 
                key
            )
            
            # if verification_passed:
            #     # 基准测试
            #     print(f"\n性能测试{model_type}...")
            #     benchmark_results = benchmark_model(model_wrapper, num_iterations=20)
                
            #     all_results[model_type] = {
            #         'verification_passed': verification_passed,
            #         'benchmark': benchmark_results,
            #         'model_params': sum(p.numel() for p in model.parameters()),
            #     }
                
            #     print(f"\n✅ {model_type}测试完成")
            # else:
            #     all_results[model_type] = {
            #         'verification_passed': False,
            #         'error': '验证失败',
            #     }
            #     print(f"\n❌ {model_type}验证失败")
                
        except Exception as e:
            print(f"❌ {model_type}测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_type] = {
                'verification_passed': False,
                'error': str(e),
            }
    
    # 生成最终报告
    # print(f"\n{'='*80}")
    # print("📋 最终测试报告")
    # print('='*80)
    
    # for model_type, results in all_results.items():
    #     print(f"\n{model_type}:")
    #     if results.get('verification_passed', False):
    #         print(f"  ✅ 验证通过")
    #         benchmark = results.get('benchmark', {})
    #         if benchmark:
    #             print(f"  性能指标:")
    #             print(f"    平均每批时间: {benchmark.get('avg_batch_time_ms', 0):.2f} ms")
    #             print(f"    平均解密时间: {benchmark.get('avg_decryption_ms', 0):.2f} ms")
    #             print(f"    平均模型时间: {benchmark.get('avg_model_ms', 0):.2f} ms")
    #             print(f"    FPS: {benchmark.get('fps', 0):.2f}")
    #         print(f"  模型参数量: {results.get('model_params', 0):,}")
    #     else:
    #         print(f"  ❌ 验证失败")
    #         if 'error' in results:
    #             print(f"    错误: {results['error']}")
    
    # # 总结
    # print(f"\n{'='*80}")
    # print("🎉 测试总结")
    # print('='*80)
    
    # passed_models = [m for m, r in all_results.items() if r.get('verification_passed', False)]
    # failed_models = [m for m, r in all_results.items() if not r.get('verification_passed', False)]
    
    # print(f"通过测试的模型: {', '.join(passed_models) if passed_models else '无'}")
    # print(f"未通过测试的模型: {', '.join(failed_models) if failed_models else '无'}")
    
    # if passed_models:
    #     print(f"\n✅ 成功在{len(passed_models)}个模型上集成AES解密算子")
    # else:
    #     print(f"\n❌ 所有模型测试均失败")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    # torch.manual_seed(42)
    # np.random.seed(42)
    main()