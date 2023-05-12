#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

from sys import platform
if platform == "darwin":
    # OS X
    os.environ["CC"] = "clang++"
    os.environ["CXX"] = "clang++"


setup(
    name='pr_cpp',
    ext_modules=[
        CppExtension('pr_cpp', ['pr.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
