/* -*- c++ -*- */

#ifndef PM_H
#define PM_H

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

//#include <omp.h>

#if defined(_USE_GPU)

#if defined(_GPU_CUDA)
#include "pm_cuda.h"
#elif defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)
#include "pm_sycl.h"
#elif defined(_GPU_OPENMP)
#error "Attempting to use -D_GPU_OPENMP which is not currently supported"
#include "pm_openmp.h"
#endif

#elif defined(_USE_CPU)

#include "pm_host.h"

#endif

#endif
