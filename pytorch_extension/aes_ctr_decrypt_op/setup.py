from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import torch_npu
import os

# --- paths ---
torch_dir = os.path.dirname(torch.__file__)
torch_npu_dir = os.path.dirname(torch_npu.__file__)

cann_dir = "/usr/local/Ascend/cann-8.5.1"
cust_op_api_dir = f"{cann_dir}/opp/vendors/custom_aes_ctr/op_api"

include_dirs = [
    f"{torch_dir}/include",
    f"{torch_npu_dir}/include",
    f"{cann_dir}/runtime/include",
    f"{cann_dir}/aarch64-linux/include",
    f"{cust_op_api_dir}/include",
]

library_dirs = [
    f"{torch_npu_dir}/lib",
    f"{cann_dir}/runtime/lib64",
    f"{cann_dir}/aarch64-linux/lib64",
    f"{cann_dir}/lib64",
    f"{cust_op_api_dir}/lib",
]

libraries = [
    "torch_npu",
    "ascendcl",
    "cust_opapi",
]

extra_compile_args = ["-std=c++17", "-O2", "-fPIC", "-Wno-sign-compare"]
extra_link_args = [
    f"-Wl,-rpath,{torch_npu_dir}/lib",
    f"-Wl,-rpath,{cust_op_api_dir}/lib",
]

ext = CppExtension(
    "aes_ctr_decrypt_op._C",
    ["aes_ctr_decrypt_kernel.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="aes_ctr_decrypt_op",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
