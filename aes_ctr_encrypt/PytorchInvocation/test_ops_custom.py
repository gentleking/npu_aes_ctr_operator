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
from torch_npu.testing.testcase import TestCase, run_tests

from Crypto.Cipher import AES
import hashlib

torch.npu.config.allow_internal_format = False

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
    
    print("golden MD5: ", golden_md5)
    print("output MD5: ", output_md5)

class TestAesCtrEncrypt(TestCase):

    def test_aes_ctr_encrypt(self):
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
        
        print("golden MD5: ", golden_md5)
        print("output MD5: ", output_md5)

if __name__ == "__main__":
    run_tests()
    test()