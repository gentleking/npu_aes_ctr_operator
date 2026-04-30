import numpy as np
from Crypto.Cipher import AES
import subprocess
import os, time, json

KEY = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10], dtype=np.uint8)
IV = np.zeros(16, dtype=np.uint8)

data_sizes = [
    1024,       # 1 KB
    4096,       # 4 KB
    16384,      # 16 KB
    65536,      # 64 KB
    262144,     # 256 KB
    1048576,    # 1 MB
    4194304,    # 4 MB
    16777216,   # 16 MB
]

base_dir = "/root/operator-dev/aes_ctr_decrypt/AclNNInvocation"
results = []

for dsize in data_sizes:
    data_len = dsize // 2  # uint16 elements
    plaintext = np.arange(data_len, dtype=np.uint16) % 65535
    cipher = AES.new(KEY.tobytes(), AES.MODE_CTR, initial_value=IV.tobytes(), nonce=b'')
    ciphertext = cipher.encrypt(plaintext.tobytes())

    # Write input files
    with open(os.path.join(base_dir, "input", "plaintext.bin"), "wb") as f:
        f.write(ciphertext)
    KEY.tofile(os.path.join(base_dir, "input", "key.bin"))
    plaintext.tofile(os.path.join(base_dir, "output", "golden.bin"))

    # Run 5 iterations
    times = []
    for i in range(5):
        start = time.time()
        result = subprocess.run(
            [os.path.join(base_dir, "output", "execute_add_op")],
            cwd=base_dir, capture_output=True, text=True, timeout=60
        )
        elapsed = time.time() - start
        times.append(elapsed)

    # Verify
    verify = subprocess.run(
        ["python3", os.path.join(base_dir, "scripts", "verify_result.py"),
         os.path.join(base_dir, "output", "cyphertext.bin"),
         os.path.join(base_dir, "output", "golden.bin")],
        capture_output=True, text=True
    )
    status = "PASS" if "matches" in verify.stdout else "FAIL"

    avg = np.mean(times[1:])  # skip first (warmup)
    results.append({
        "data_size_bytes": dsize,
        "data_size_str": f"{dsize/1024:.0f} KB" if dsize < 1024*1024 else f"{dsize/1024/1024:.0f} MB",
        "avg_time_ms": round(avg * 1000, 2),
        "throughput_MBs": round(dsize / avg / 1024 / 1024, 2),
        "status": status
    })
    print(f"{results[-1]['data_size_str']:>8s} | avg: {results[-1]['avg_time_ms']:>8.2f} ms | "
          f"throughput: {results[-1]['throughput_MBs']:>8.2f} MB/s | {status}")

print("\n" + "="*60)
for r in results:
    print(f"{r['data_size_str']:>8s} | {r['avg_time_ms']:>8.2f} ms | {r['throughput_MBs']:>8.2f} MB/s | {r['status']}")
