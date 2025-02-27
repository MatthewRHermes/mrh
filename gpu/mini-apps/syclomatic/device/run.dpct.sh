cp ../../../src/pm/device.h ./
cp ../../../src/pm/device_cuda.cpp ./

rm -rf dpct_output
dpct --extra-arg="-D_GPU_CUDA" device_cuda.cpp
