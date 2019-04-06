import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ["src/cpu.c"]
headers = ["src/cpu.h"]
defines = []
with_cuda = False

if torch.cuda.is_available():
    print("Including CUDA code.")
    sources += ["src/gpu.c"]
    headers += ["src/gpu.h"]
    defines += [("WITH_CUDA", None)]
    with_cuda = True
else: print("No CUDA....")

this_file = os.path.dirname(os.path.realpath(__file__))
# print(this_file)
extra_objects = ["src/gpu_kernel.o"]
extra_objects = [os.path.join(this_file,fname) for fname in extra_objects]

ffi = create_extension(
    "meshrender",
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=["-std=c99"],
    extra_objects=extra_objects,
)

if __name__ == "__main__":
    ffi.build()
