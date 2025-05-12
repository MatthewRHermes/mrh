/* -*- c++ -*- */

#ifndef MATHLIB_H
#define MATHLIB_H

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

#if defined(_USE_GPU)

// set default based on backend if one not explicitly set
#if !defined(_GPU_CUBLAS) && !defined(_GPU_MKL) && !defined(_GPU_HIPBLAS)

#if defined(_GPU_CUDA) || defined(_GPU_SYCL_CUDA)
#error "Did you forget to set -D_GPU_CUBLAS?"
#elif defined(_GPU_SYCL)
#error "Did you forget to set -D_GPU_MKL?"
#elif defined(_GPU_HIP)
#error "Did you forget to set -D_GPU_HIPBLAS?"
#endif

#endif

// load appropriate header

#if defined(_GPU_CUBLAS)
#include "mathlib_cublas.h"
#elif defined(_GPU_MKL)
#include "mathlib_mkl.h"
#elif defined(_GPU_HIPBLAS)
#include "mathlib_hipblas.h"
#endif

#elif defined(_USE_CPU)

#include "mathlib_host.h"

#endif

#endif
