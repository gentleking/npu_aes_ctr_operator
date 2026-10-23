import os
import torch
import torch_npu

lib_path = os.path.join(os.path.dirname(__file__), "_C.cpython-311-aarch64-linux-gnu.so")
torch.ops.load_library(lib_path)

npu_aes_ctr_decrypt = torch.ops.aes_ctr.decrypt
