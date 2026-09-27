from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


file_path = "utils/gs_cuda_dmax"

setup(
    name="gscuda_batch",
    ext_modules=[
        CUDAExtension(
            name="gscuda_batch",
            sources=[
                os.path.join(file_path, "gswrapper_batch.cpp"),
                os.path.join(file_path, "gs_batch.cu")
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)