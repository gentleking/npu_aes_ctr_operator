#!/root/miniconda3/envs/operator-dev/bin/python3
import torch
import torch_npu
import time
import hashlib
import numpy as np
from Crypto.Cipher import AES

torch_npu.npu.set_compile_mode(jit_compile=False)

KEY = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
IV = torch.zeros((1, 16), dtype=torch.uint8)
WARMUP = 3
ITERS = 5

sizes = [
    (1, "1M"),
    (2, "2M"),
    (4, "4M"),
    (8, "8M"),
    (16, "16M"),
    (32, "32M"),
    (64, "64M"),
    (128, "128M"),
    (256, "256M"),
    (512, "512M"),
    (1024, "1G"),
    (2048, "2G"),
]

results = []

print(f"{'Size':>8s} {'Bytes':>12s} {'Avg(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s} {'MB/s':>10s} {'GB/s':>10s}")
print("-" * 70)

for size_mb, label in sizes:
    num_bytes = size_mb * 1024 * 1024
    num_uint8 = num_bytes  # uint8 = 1 byte per element

    plaintext = torch.randint(0, 256, size=(num_uint8,), dtype=torch.uint8)
    cipher = AES.new(KEY.numpy().tobytes(), AES.MODE_CTR, initial_value=IV.numpy().tobytes(), nonce=b'')
    _golden = torch.frombuffer(bytearray(cipher.encrypt(plaintext.numpy().tobytes())), dtype=torch.uint8)

    torch.npu.synchronize()
    plain_npu = plaintext.npu()
    key_npu = KEY.npu()
    torch.npu.synchronize()

    for warm in range(WARMUP):
        out = torch_npu.npu_aes_ctr_decrypt(plain_npu, key_npu)
        torch.npu.synchronize()
        del out

    timings = []
    for i in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = torch_npu.npu_aes_ctr_decrypt(plain_npu, key_npu)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        timings.append(elapsed)

        if i == 0:
            out_md5 = hashlib.md5(out.cpu().numpy().tobytes()).hexdigest()
            golden_md5 = hashlib.md5(_golden.numpy().tobytes()).hexdigest()
            assert out_md5 == golden_md5, f"FAIL at {label}: MD5 mismatch"

        del out

    avg_ms = np.mean(timings)
    min_ms = np.min(timings)
    max_ms = np.max(timings)
    throughput_mb = (num_bytes / 1024 / 1024) / (avg_ms / 1000)
    throughput_gb = throughput_mb / 1024

    results.append((label, num_bytes, avg_ms, min_ms, max_ms, throughput_mb, throughput_gb))
    print(f"{label:>8s} {num_bytes:>12d} {avg_ms:>10.2f} {min_ms:>10.2f} {max_ms:>10.2f} {throughput_mb:>10.2f} {throughput_gb:>10.3f}")

    del plaintext, plain_npu, key_npu
    torch.npu.empty_cache()

print("\n" + "=" * 70)
print("SUMMARY (size, bytes, avg_ms, min_ms, max_ms, MB/s, GB/s)")
for r in results:
    print(f"{r[0]:>8s} {r[1]:>12d} {r[2]:>10.2f} {r[3]:>10.2f} {r[4]:>10.2f} {r[5]:>10.2f} {r[6]:>10.3f}")
