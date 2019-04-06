#!/bin/sh
CUDA_PATH=/usr/local/cuda
$CUDA_PATH/bin/nvcc -c -o src/gpu_kernel.o src/gpu_kernel.cu -x cu -Xcompiler -fPIC #-arch=sm_61
python3 build.py
mv meshrender/* .
rmdir meshrender

