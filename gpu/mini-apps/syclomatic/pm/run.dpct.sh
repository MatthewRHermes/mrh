#cp ../../../src/pm/pm_cuda.cpp ./
#cp ../../../src/pm/pm_cuda.h ./

rm -rf dpct_output
dpct --extra-arg="-D_USE_GPU" --extra-arg="-D_GPU_CUDA" pm_cuda.cpp
