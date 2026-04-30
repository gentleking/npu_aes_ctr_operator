#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

from typing import Any
import torch
from torch.library import impl
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
from torch_npu.op_plugin.meta._meta_registrations import m

from Crypto.Cipher import AES
import hashlib


@impl(m, "npu_aes_ctr_encrypt")
def npu_aes_ctr_encrypt_meta(plaintext, key):
    return torch.empty_like(plaintext)


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.npu.npu_aes_ctr_encrypt.default)
def convert_npu_aes_ctr_encrypt(plaintext: Tensor, key: Tensor, cyphertext: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "AesCtrEncrypt",
        inputs={
            "plaintext": plaintext,
            "key": key,
        },
        outputs=['cyphertext']
    )


class TestTorchCompileAesCtrEncrypt(TestCase):

    def test_aes_ctr_encrypt(self):
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        # 生成plaintext: [8, 2048], uint8
        plaintext = torch.randint(0, 256, size=(8, 2048), dtype=torch.uint8)

        # 生成key: [1, 16], uint8
        key = torch.randint(0, 256, size=(1, 16), dtype=torch.uint8)
        print(plaintext, '\n', key)
        # length = [8, 2048]
        # x = torch.rand(length, device='npu', dtype=torch.uint8)
        # y = torch.rand(length, device='npu', dtype=torch.uint8)
        # print(x, '\n', y)
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, plaintext, key):
                return torch_npu.npu_aes_ctr_encrypt(plaintext, key)
        mod = torch.compile(Module().npu(), backend=npu_backend)
        output = mod(plaintext, key)
        # print(output)
        # self.assertRtolEqual(output, (x + y))

        print("Cyphertext:", output)
        
        # 生成全0的 iv，形状 [1, 16]，类型 uint8
        iv = torch.zeros((1, 16), dtype=torch.uint8)
        cipher = AES.new(key.numpy().tobytes(), AES.MODE_CTR, initial_value=iv.numpy().tobytes(), nonce=b'')
        golden = cipher.encrypt(plaintext.numpy().tobytes())

        output_md5 = hashlib.md5(output.numpy().tobytes()).hexdigest()
        golden_md5 = hashlib.md5(golden).hexdigest()
        is_match = golden_md5 == output_md5
        
        print("golden output MD5: ", golden_md5)
        print("cypher output MD5: ", output_md5)

if __name__ == "__main__":
    run_tests()
