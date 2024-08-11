"""
    Copyright (c) 2024, Fireworks AI.
"""

from setuptools import setup
from torch.utils import cpp_extension

setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="amd_repro.lib",
            sources=[
                "amd_repro/registry.cpp",
                "amd_repro/virtual_memory.cpp",
                "amd_repro/multi_gpu_barrier.hip",
            ],
            include_dirs=[
                "amd_repro",
            ],
            extra_compile_args={
                "cxx": [
                    "-std=c++17",
                    "-O3",
                ],
                "nvcc": [
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-g",
                    # hip-specific
                    "-save-temps",
                    "-U__HIP_NO_HALF_CONVERSIONS__",
                    "-U__HIP_NO_HALF_OPERATORS__",
                    "-U__HIP_NO_HALF2_OPERATORS__",
                    "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
    },
)
