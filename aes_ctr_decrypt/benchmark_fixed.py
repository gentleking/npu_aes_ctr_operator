import numpy as np
from Crypto.Cipher import AES
import subprocess, os

base_dir = "/root/operator-dev/aes_ctr_decrypt/AclNNInvocation"
KEY = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10], dtype=np.uint8)
IV = np.zeros(16, dtype=np.uint8)

dsize = 8192 * 2  # 16384 bytes (which matches {8,2048} uint8 = 16384)
data_len = dsize // 2

plaintext = np.arange(data_len, dtype=np.uint16) % 65535
cipher = AES.new(KEY.tobytes(), AES.MODE_CTR, initial_value=IV.tobytes(), nonce=b'')
ciphertext = cipher.encrypt(plaintext.tobytes())

with open(os.path.join(base_dir, "input", "plaintext.bin"), "wb") as f:
    f.write(ciphertext)
KEY.tofile(os.path.join(base_dir, "input", "key.bin"))
plaintext.tofile(os.path.join(base_dir, "output", "golden.bin"))

# Warmup
subprocess.run([os.path.join(base_dir, "output", "execute_add_op")], cwd=base_dir, capture_output=True)

import time
times = []
for i in range(10):
    start = time.perf_counter()
    subprocess.run([os.path.join(base_dir, "output", "execute_add_op")], cwd=base_dir, capture_output=True)
    times.append(time.perf_counter() - start)

avg_s = np.mean(times)
print(f"\nData size: {dsize} bytes (16 KB)")
print(f"Average time: {avg_s*1000:.2f} ms over {len(times)} runs")
print(f"Throughput: {dsize/avg_s/1024/1024:.2f} MB/s")

result = subprocess.run(
    ["python3", os.path.join(base_dir, "scripts", "verify_result.py"),
     os.path.join(base_dir, "output", "cyphertext.bin"),
     os.path.join(base_dir, "output", "golden.bin")],
    capture_output=True, text=True
)
print(f"Verification: {'PASS' if 'matches' in result.stdout else 'FAIL'}")
