/* -*- c++ -*- */

#ifndef MATHLIB_H
#define MATHLIB_H

#if defined(_USE_GPU)

#if defined(_GPU_CUDA) || defined(_GPU_SYCL_CUDA)
#include "mathlib_cuda.h"

#elif defined(_GPU_SYCL_ONEAPI)
#include "mathlib_mkl.h"

#endif

#elif defined(_USE_CPU)

#include "mathlib_host.h"

#endif

#endif
