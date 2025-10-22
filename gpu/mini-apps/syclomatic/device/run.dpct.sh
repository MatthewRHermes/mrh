#cp ../../../src/device.h ./
#cp ../../../src/pm/device_cuda.cpp ./

rm -rf dpct_output
OPTIONS="--extra-arg=\"-D_GPU_CUDA\" --cuda-include-path=$NVHPC_CUDA_HOME/../include "

dpct --extra-arg="-D_GPU_CUDA" --cuda-include-path=$NVHPC_CUDA_HOME/../include  device_cuda.cpp
