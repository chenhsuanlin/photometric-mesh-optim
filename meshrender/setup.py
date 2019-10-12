from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CppExtension,CUDAExtension

setup(
    name="meshrender",
    ext_modules=[
        CUDAExtension(
            name="meshrender",
            sources=["src/rasterize.cpp","src/rasterize_cuda.cu"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
