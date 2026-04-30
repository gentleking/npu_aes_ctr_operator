#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu

from Crypto.Cipher import AES
import hashlib

# torch.npu.config.allow_internal_format = False

# torch_npu.npu.set_compile_mode(jit_compile=False)

def test():
    # 生成plaintext: [8, 2048], uint8
    plaintext = torch.randint(0, 256, size=(8, 2048), dtype=torch.uint8)

    # 生成key: [1, 16], uint8
    key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
    # print(plaintext, '\n', key)
    print("plaintext: ", plaintext, '\n', "plaintext size: ", plaintext.size())
    print("key: ", key, '\n', "key size: ", key.size())

    torch.npu.synchronize()
    output = torch_npu.npu_aes_ctr_encrypt(plaintext.npu(), key.npu()).cpu()
    torch.npu.synchronize()

    print("Cyphertext:", output, '\n', 'cyphertext size: ', output.size())
    
    # 生成全0的 iv，形状 [1, 16]，类型 uint8
    iv = torch.zeros((1, 16), dtype=torch.uint8)
    cipher = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
    golden = cipher.encrypt(plaintext.numpy().tobytes())

    output_md5 = hashlib.md5(output.numpy().tobytes()).hexdigest()
    golden_md5 = hashlib.md5(golden).hexdigest()
    is_match = golden_md5 == output_md5
    print(f"match result: {'OK' if is_match else 'Failed'}')")
    print("golden MD5: ", golden_md5)
    print("output MD5: ", output_md5)

def test_add():
    x1 = torch.randint(0, 256, size=(8, 2048), dtype=torch.uint8).npu()
    x2 = torch.randint(0, 256, size=(8, 2048), dtype=torch.uint8).npu()

    torch.npu.synchronize()
    output = x1 + x2
    torch.npu.synchronize()

    print("Output:", output, '\n', 'output size: ', output.size())


if __name__ == "__main__":
    experimental_config_ = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2
    )
    
    prof = torch_npu.profiler.profile(
        experimental_config=experimental_config_,
        with_stack=True,
    )
    
    prof.start()
    test()
    test_add()
    prof.stop()
    prof.export_chrome_trace('./trace/chrome_trace_1.json')