"""
AES-CTR 加解密接口测试脚本（NPU 自定义算子版）。

注意：算子要求数据至少 128 float16 元素（256 字节），
因为 AI Core kernel 需要至少 2 个 tile 才能运行流水线。
"""

import sys
import os

sys.path.insert(0, "/workspace/LMCache-Ascend")
sys.path.insert(0, "/workspace/ascend-op-dev/pytorch_extension")

_cust_api_lib = "/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_aes_ctr/op_api/lib"
_curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _cust_api_lib not in _curr_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_cust_api_lib}:{_curr_ld}"

import torch
import torch_npu
from lmcache_ascend.v1.crypto import aes_ctr_encrypt


_MIN_ELEMS = 128  # 最小元素数（float16），= 2 tiles of 64 halfs


def test_roundtrip():
    key = torch.randint(0, 256, (16,), dtype=torch.uint8, device="npu")
    plain = torch.randn(_MIN_ELEMS, dtype=torch.float16, device="npu")
    orig = plain.clone()

    cipher = aes_ctr_encrypt(plain.clone(), key)
    assert not torch.allclose(orig.cpu(), cipher.cpu()), "Cipher should differ from plain"

    decrypted = aes_ctr_encrypt(cipher.clone(), key)
    assert torch.allclose(orig.cpu(), decrypted.cpu(), atol=1e-3), "Roundtrip failed"
    print(f"  [PASS] roundtrip: orig={orig.cpu()[0]:.4f} -> decrypted={decrypted.cpu()[0]:.4f}")


def test_different_key_produces_different_output():
    key1 = torch.full((16,), 1, dtype=torch.uint8, device="npu")
    key2 = torch.full((16,), 2, dtype=torch.uint8, device="npu")
    data = torch.randn(_MIN_ELEMS, dtype=torch.float16, device="npu")

    c1 = aes_ctr_encrypt(data.clone(), key1)
    c2 = aes_ctr_encrypt(data.clone(), key2)
    assert not torch.allclose(c1.cpu(), c2.cpu()), "Different keys should give different cipher"
    print("  [PASS] different keys -> different cipher")


def test_inplace():
    key = torch.randint(0, 256, (16,), dtype=torch.uint8, device="npu")
    data = torch.randn(_MIN_ELEMS, dtype=torch.float16, device="npu")
    data_ptr = data.data_ptr()

    result = aes_ctr_encrypt(data, key)
    assert result.data_ptr() == data_ptr, "Should modify in-place"
    print("  [PASS] in-place modification")


def test_various_shapes():
    key = torch.randint(0, 256, (16,), dtype=torch.uint8, device="npu")

    configs = [
        (torch.Size([2, 28, 256, 256]), torch.float16),   # KV cache chunk
        (torch.Size([128]), torch.float16),                 # 1D (min size)
        (torch.Size([16, 16]), torch.float32),              # 2D (256 float32 = 1024 bytes)
        (torch.Size([256]), torch.uint8),                   # bytes
    ]

    for shape, dtype in configs:
        if dtype == torch.uint8:
            data = torch.randint(0, 256, shape, dtype=dtype, device="npu")
        else:
            data = torch.randn(shape, dtype=dtype, device="npu")
        orig = data.clone()
        aes_ctr_encrypt(data, key)
        aes_ctr_encrypt(data, key)
        match = torch.allclose(orig.cpu().float(), data.cpu().float(), atol=1e-3) if dtype != torch.uint8 else torch.equal(orig.cpu(), data.cpu())
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] shape={shape}, dtype={dtype}")
        assert match, f"Roundtrip failed for shape={shape}, dtype={dtype}"


if __name__ == "__main__":
    print("=== AES-CTR Encrypt Tests (NPU Custom Op) ===\n")

    tests = [
        ("roundtrip", test_roundtrip),
        ("different key", test_different_key_produces_different_output),
        ("in-place", test_inplace),
        ("various shapes", test_various_shapes),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[{passed}/{len(tests)}] passed")
    sys.exit(0 if passed == len(tests) else 1)
