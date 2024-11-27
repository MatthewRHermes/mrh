# cp ../../../../src/device.h ./
# cp ../../../../src/device_cuda.cpp ./
# comment out preprocessor to expose code (must be flag to pass -D_GPU_CUDA...)

rm -r dpct_output
dpct device_cuda.cpp
