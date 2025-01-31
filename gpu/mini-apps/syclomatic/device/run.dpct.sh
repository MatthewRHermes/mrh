cp ../../../src/device.h ./
cp ../../../src/device_cuda.cpp ./

rm -rf dpct_output
dpct --extra-arg="-D_GPU_CUDA" device_cuda.cpp
