#cp ../../src/pm/pm.h ./
#cp ../../src/pm/pm_cuda.h ./
#cp ../../src/pm/pm_cuda.cpp ./
#
#cp ../../src/device.h ./
#cp ../../src/pm/device_cuda.cpp ./

# cleanup omp.h and #if 1 pragmas...

#../../../../hipify/dist/bin/hipify-clang pm_cuda.cpp -- -D_USE_GPU -D_GPU_CUDA
#../../../../hipify/dist/bin/hipify-clang pm_cuda.h -- -D_USE_GPU -D_GPU_CUDA

PYTHON_INC=`python -m pybind11 --includes`

# cleanup #if 1 pragmas
#../../../../hipify/dist/bin/hipify-clang device_cuda.cpp -- -D_USE_GPU -D_GPU_CUDA -D_GPU_CUBLAS ${PYTHON_INC}


# cleanup single_precision and cublas_vs header
#../../../../hipify/dist/bin/hipify-clang mathlib_cublas.cpp -- -D_USE_GPU -D_GPU_CUDA -D_GPU_CUBLAS
