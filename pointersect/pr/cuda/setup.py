#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

from sys import platform
if platform == "darwin":
    # OS X
    os.environ["CC"] = "clang++"
    os.environ["CXX"] = "clang++"


setup(
    name='pr_cuda',
    ext_modules=[
        CUDAExtension('pr_cuda', [
            'pr_cuda.cpp',
            'pr_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
