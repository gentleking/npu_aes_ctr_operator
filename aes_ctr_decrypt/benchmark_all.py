#!/usr/bin/env python3
import os, sys, time, hashlib, subprocess
import numpy as np
from Crypto.Cipher import AES

KEY = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10], dtype=np.uint8)
IV = np.zeros(16, dtype=np.uint8)

HAS_TORCH_NPU = False
try:
    import torch
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    HAS_TORCH_NPU = True
except ImportError:
    pass

ACLNN_EXEC = "/root/operator-dev/aes_ctr_decrypt/AclNNInvocation/output/execute_add_op"
ACLNN_DIR = "/root/operator-dev/aes_ctr_decrypt/AclNNInvocation"
ACLNN_VERIFY = os.path.join(ACLNN_DIR, "scripts", "verify_result.py")
HAS_ACLNN = os.path.isfile(ACLNN_EXEC)

if not HAS_TORCH_NPU and not HAS_ACLNN:
    print("No test backend available: torch_npu not found and ACLNN executable not built")
    sys.exit(1)

WARMUP = 3
ITERS = 5

data_sizes = [
    (1024, "1KB"),
    (4096, "4KB"),
    (16384, "16KB"),
    (65536, "64KB"),
    (262144, "256KB"),
    (1048576, "1M"),
    (2097152, "2M"),
    (4194304, "4M"),
    (8388608, "8M"),
    (16777216, "16M"),
    (33554432, "32M"),
    (67108864, "64M"),
    (134217728, "128M"),
    (268435456, "256M"),
    (536870912, "512M"),
    (1073741824, "1G"),
    (2147483648, "2G"),
]

def run_aclnn(dsize):
    data_len = dsize // 2
    plaintext = np.arange(data_len, dtype=np.uint16) % 65535
    cipher = AES.new(KEY.tobytes(), AES.MODE_CTR, initial_value=IV.tobytes(), nonce=b'')
    ciphertext = cipher.encrypt(plaintext.tobytes())
    golden_bytes = plaintext.tobytes()

    with open(os.path.join(ACLNN_DIR, "input", "plaintext.bin"), "wb") as f:
        f.write(ciphertext)
    KEY.tofile(os.path.join(ACLNN_DIR, "input", "key.bin"))
    with open(os.path.join(ACLNN_DIR, "output", "golden.bin"), "wb") as f:
        f.write(golden_bytes)

    timings = []
    for i in range(WARMUP + ITERS):
        start = time.perf_counter()
        result = subprocess.run([ACLNN_EXEC], cwd=ACLNN_DIR, capture_output=True, timeout=120)
        elapsed = time.perf_counter() - start
        if i >= WARMUP:
            timings.append(elapsed)
        if result.returncode != 0:
            print(f"  ERROR: ACLNN exec failed (size={dsize})")
            return None, None, "FAIL"

    verify = subprocess.run(
        ["python3", ACLNN_VERIFY,
         os.path.join(ACLNN_DIR, "output", "cyphertext.bin"),
         os.path.join(ACLNN_DIR, "output", "golden.bin")],
        capture_output=True, text=True
    )
    status = "PASS" if "matches" in verify.stdout else "FAIL"
    return timings, _golden_bytes_to_md5(golden_bytes), status  # noqa

def _golden_bytes_to_md5(b):
    return hashlib.md5(b).hexdigest()

def run_torch(dsize):
    torch_npu = sys.modules["torch_npu"]
    torch = sys.modules["torch"]

    plaintext = torch.randint(0, 256, size=(dsize,), dtype=torch.uint8)
    key_t = torch.from_numpy(KEY.copy()).unsqueeze(0)
    cipher = AES.new(KEY.tobytes(), AES.MODE_CTR, initial_value=IV.tobytes(), nonce=b'')
    golden_md5 = hashlib.md5(cipher.encrypt(plaintext.numpy().tobytes())).hexdigest()

    torch.npu.synchronize()
    plain_npu = plaintext.npu()
    key_npu = key_t.npu()
    torch.npu.synchronize()

    for _ in range(WARMUP):
        out = torch_npu.npu_aes_ctr_decrypt(plain_npu, key_npu)
        torch.npu.synchronize()
        del out

    timings = []
    status = "PASS"
    for i in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = torch_npu.npu_aes_ctr_decrypt(plain_npu, key_npu)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

        if i == 0:
            out_md5 = hashlib.md5(out.cpu().numpy().tobytes()).hexdigest()
            if out_md5 != golden_md5:
                status = "FAIL"
        del out

    del plaintext, plain_npu, key_npu
    if HAS_TORCH_NPU:
        torch_npu = sys.modules["torch_npu"]
        torch_npu.npu.empty_cache()

    return timings, golden_md5, status

print(f"{'Size':>8s} {'Bytes':>12s} {'Avg(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s} {'MB/s':>10s} ")
print("-" * 72)

results = []

for dsize, label in data_sizes:
    if HAS_TORCH_NPU:
        timings, golden_md5, status = run_torch(dsize)
    elif HAS_ACLNN:
        timings, golden_md5, status = run_aclnn(dsize)
    else:
        print(f"{label:>8s}  no backend available")
        continue

    if timings is None:
        results.append((label, dsize, 0, 0, 0, 0, status))
        print(f"{label:>8s} {dsize:>12d} {'ERROR':>10s}")
        continue
        continue

    avg_ms = np.mean(timings)
    min_ms = np.min(timings)
    max_ms = np.max(timings)
    throughput_mb = (dsize / 1024 / 1024) / (avg_ms / 1000)

    results.append((label, dsize, avg_ms, min_ms, max_ms, throughput_mb, status))
    print(f"{label:>8s} {dsize:>12d} {avg_ms:>10.2f} {min_ms:>10.2f} {max_ms:>10.2f} {throughput_mb:>10.2f} {status}")

print("\n" + "=" * 72)
print("SUMMARY")
print(f"{'Size':>8s} {'Bytes':>12s} {'Avg(ms)':>10s} {'MB/s':>10s} {'Status':>8s}")
print("-" * 52)
for r in results:
    label, dsize, avg_ms, _, _, throughput_mb, status = r
    if avg_ms == 0:
        print(f"{label:>8s} {dsize:>12d} {'FAIL':>10s}")
    else:
        print(f"{label:>8s} {dsize:>12d} {avg_ms:>10.2f} {throughput_mb:>10.2f} {status:>8s}")
