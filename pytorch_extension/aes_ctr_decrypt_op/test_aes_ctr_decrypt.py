import torch
import torch_npu
from Crypto.Cipher import AES
import hashlib

from aes_ctr_decrypt_op import npu_aes_ctr_decrypt

def test_aes_ctr_decrypt():
    print("=" * 60)
    print("AES-CTR Decrypt Custom Op Test")
    print("=" * 60)

    plaintext = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
    key = torch.randint(0, 256, (1, 16), dtype=torch.uint8)
    iv = torch.zeros((1, 16), dtype=torch.uint8)

    print(f"\n[1] Plaintext shape: {plaintext.shape}, dtype: {plaintext.dtype}")
    print(f"    Key shape: {key.shape}, dtype: {key.dtype}")

    print("\n[2] Encrypting with pycryptodome (CPU)...")
    cipher = AES.new(key.numpy().tobytes(), AES.MODE_CTR,
                     initial_value=iv.numpy().tobytes(), nonce=b'')
    ciphertext = torch.frombuffer(
        bytearray(cipher.encrypt(plaintext.numpy().tobytes())),
        dtype=torch.uint8
    ).reshape(plaintext.shape)

    print("\n[3] Decrypting on NPU (custom op)...")
    cipher_npu = ciphertext.npu()
    key_npu = key.npu()

    torch.npu.synchronize()
    decrypted = npu_aes_ctr_decrypt(cipher_npu, key_npu)
    torch.npu.synchronize()
    decrypted_cpu = decrypted.cpu()

    print("\n[4] Verifying...")
    plain_md5 = hashlib.md5(plaintext.numpy().tobytes()).hexdigest()
    decrypted_md5 = hashlib.md5(decrypted_cpu.numpy().tobytes()).hexdigest()
    match = torch.equal(plaintext, decrypted_cpu)

    print(f"    Plaintext MD5:    {plain_md5}")
    print(f"    Decrypted MD5:    {decrypted_md5}")
    print(f"    Exact match:      {match}")

    if not match:
        raise AssertionError("Decryption result does not match plaintext")

    adder = torch.ones_like(decrypted_cpu, dtype=torch.uint8)
    result = (decrypted + adder.npu()).cpu()
    expected = plaintext + adder
    assert torch.equal(result, expected), "Post-decrypt tensor op failed"

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_aes_ctr_decrypt()
