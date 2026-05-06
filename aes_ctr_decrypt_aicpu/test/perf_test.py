import numpy as np
import subprocess
import sys
import os
import json
from Crypto.Cipher import AES

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(TEST_DIR, "input")
OUTPUT_DIR = os.path.join(TEST_DIR, "output")
MODEL_DIR = os.path.join(TEST_DIR, "op_models")

DDK_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
FIXED_KEY = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                      0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10], dtype=np.uint8)
FIXED_IV = np.zeros(16, dtype=np.uint8)

SIZES = [
    ("1M", 1 * 1024 * 1024),
    ("2M", 2 * 1024 * 1024),
    ("4M", 4 * 1024 * 1024),
    ("8M", 8 * 1024 * 1024),
    ("16M", 16 * 1024 * 1024),
    ("32M", 32 * 1024 * 1024),
    ("64M", 64 * 1024 * 1024),
    ("128M", 128 * 1024 * 1024),
    ("256M", 256 * 1024 * 1024),
    ("512M", 512 * 1024 * 1024),
    ("1G", 1024 * 1024 * 1024),
    ("2G", 2048 * 1024 * 1024),
]

NUM_WARMUP = 3
NUM_RUNS = 5
CHUNK_SIZE = 256 * 1024 * 1024

ENV = os.environ.copy()
ENV.update({
    "PATH": f"/root/miniconda3/envs/operator-dev/bin:{DDK_PATH}/bin:{ENV.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{DDK_PATH}/runtime/lib64:{DDK_PATH}/lib64:{ENV.get('LD_LIBRARY_PATH', '')}",
    "PYTHONPATH": f"{DDK_PATH}/python/site-packages",
    "ASCEND_OPP_PATH": f"{DDK_PATH}/opp",
    "ASCEND_CUSTOM_OPP_PATH": f"{DDK_PATH}/opp/vendors/customize_aicpu",
    "LC_ALL": "C",
})


def gen_op_json(data_size, filepath):
    op_json = [{
        "op": "AesCtrDecryptAicpu",
        "input_desc": [
            {"format": "ND", "shape": [data_size], "type": "uint8"},
            {"format": "ND", "shape": [16], "type": "uint8"}
        ],
        "output_desc": [
            {"format": "ND", "shape": [data_size], "type": "uint8"}
        ]
    }]
    with open(filepath, "w") as f:
        json.dump(op_json, f, indent=2)


def run_atc(op_json_path):
    cmd = ["atc", "--singleop", op_json_path, "--soc_version=Ascend310P3", "--output", MODEL_DIR]
    result = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if result.returncode != 0:
        print(f"[ERROR] ATC failed: {result.stderr[:500]}")
        return False
    return True


def run_perf_binary(data_size, warmup=3, runs=5, device=0):
    cmd = [os.path.join(TEST_DIR, "test_aes_ctr_perf"),
           str(data_size), str(warmup), str(runs), MODEL_DIR, str(device)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd=TEST_DIR)
    if result.returncode != 0:
        print(f"[ERROR] Perf binary failed: {result.stderr[:200]}")
        return None
    for line in result.stdout.strip().split("\n"):
        if line.startswith("DATA_SIZE="):
            parts = line.split()
            parsed = {}
            for p in parts:
                k, v = p.split("=")
                if k in ("DATA_SIZE", "RUNS"):
                    parsed[k] = int(v)
                else:
                    parsed[k] = float(v)
            return parsed
    return None


def run_chunked(plaintext, ciphertext, key_bytes, chunk_size, warmup, runs):
    total = len(plaintext)
    num_chunks = (total + chunk_size - 1) // chunk_size
    result = np.empty(total, dtype=np.uint8)
    total_time = 0.0
    all_times = []

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, total)
        cur_size = end - start

        op_json = os.path.join(TEST_DIR, f"_op_chunk_{c}.json")
        gen_op_json(cur_size, op_json)
        if not run_atc(op_json):
            return None
        os.remove(op_json)

        chunk_cipher = os.path.join(INPUT_DIR, f"_cipher_{c}.bin")
        chunk_key = os.path.join(INPUT_DIR, f"_key_{c}.bin")
        chunk_result = os.path.join(OUTPUT_DIR, f"_result_{c}.bin")

        with open(chunk_cipher, "wb") as f:
            f.write(ciphertext[start:end])
        with open(chunk_key, "wb") as f:
            f.write(key_bytes)

        result_data = run_perf_binary(cur_size, warmup, runs, device=0)
        if result_data is None:
            return None

        with open(chunk_result, "rb") as f:
            result[start:end] = np.frombuffer(f.read(), dtype=np.uint8)

        os.remove(chunk_cipher)
        os.remove(chunk_key)
        os.remove(chunk_result)

        total_time += result_data["AVG_MS"]
        all_times.append(result_data["AVG_MS"])

        om_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".om")]
        for f in om_files:
            os.remove(os.path.join(MODEL_DIR, f))

    avg_chunk = sum(all_times) / len(all_times)
    return {
        "AVG_MS": total_time,
        "MIN_MS": min(all_times),
        "MAX_MS": max(all_times),
        "THROUGHPUT_MBPS": (total / 1024.0 / 1024.0) / (total_time / 1000.0),
        "result": result,
    }


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    results_table = []
    for label, size in SIZES:
        print(f"\n{'='*70}")
        print(f"Testing {label} ({size} bytes)...")
        print(f"{'='*70}")

        print("  Generating test data...")
        plaintext = np.arange(size, dtype=np.uint8)
        cipher = AES.new(FIXED_KEY.tobytes(), AES.MODE_CTR,
                         initial_value=FIXED_IV.tobytes(), nonce=b'')
        ciphertext = cipher.encrypt(plaintext.tobytes())
        key_bytes = FIXED_KEY.tobytes()

        op_json_path = os.path.join(TEST_DIR, f"_op_{label}.json")
        gen_op_json(size, op_json_path)

        print("  Running ATC...")
        if not run_atc(op_json_path):
            print(f"  [SKIP] ATC failed for {label}")
            os.remove(op_json_path)
            continue
        os.remove(op_json_path)

        need_chunked = size > 512 * 1024 * 1024

        if need_chunked:
            print(f"  Executing in chunks of {CHUNK_SIZE // (1024*1024)}MB...")
            result = run_chunked(plaintext, ciphertext, key_bytes,
                                 CHUNK_SIZE, NUM_WARMUP, NUM_RUNS)
            if result is None:
                print(f"  [SKIP] Execution failed for {label}")
                continue
            decrypted = result["result"]
            passed = np.array_equal(decrypted, plaintext)
            avg_ms = result["AVG_MS"]
            min_ms = result["MIN_MS"]
            max_ms = result["MAX_MS"]
            mbps = result["THROUGHPUT_MBPS"]
        else:
            print("  Executing...")
            perf_data = run_perf_binary(size, NUM_WARMUP, NUM_RUNS, device=0)
            if perf_data is None:
                print(f"  [SKIP] Execution failed for {label}")
                continue

            result_file = os.path.join(OUTPUT_DIR, "result.bin")
            decrypted = np.fromfile(result_file, dtype=np.uint8)
            passed = np.array_equal(decrypted, plaintext)
            avg_ms = perf_data["AVG_MS"]
            min_ms = perf_data["MIN_MS"]
            max_ms = perf_data["MAX_MS"]
            mbps = perf_data["THROUGHPUT_MBPS"]

        status = "PASS" if passed else "FAIL"
        print(f"  Result: {status} | Avg: {avg_ms:.1f}ms | {mbps:.2f} MB/s")
        results_table.append((label, size, avg_ms, min_ms, max_ms, mbps, status))

        om_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".om")]
        for f in om_files:
            os.remove(os.path.join(MODEL_DIR, f))

    print(f"\n\n{'='*70}")
    print("PERFORMANCE TEST RESULTS")
    print(f"{'='*70}")
    print(f"{'Label':>8} {'Size(B)':>12} {'Avg(ms)':>12} {'Min(ms)':>10} {'Max(ms)':>10} {'MB/s':>10} {'Result':>8}")
    print("-" * 72)
    for label, size, avg, mn, mx, mbps, status in results_table:
        print(f"{label:>8} {size:>12} {avg:>12.1f} {mn:>10.1f} {mx:>10.1f} {mbps:>10.2f} {status:>8}")


if __name__ == "__main__":
    main()
