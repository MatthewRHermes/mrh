# cp ../../transpose/offload/offload_cuda.cpp ./
# strip down file to just calling functions and cuda kernels

rm -r dpct_output
dpct offload_cuda.cpp
