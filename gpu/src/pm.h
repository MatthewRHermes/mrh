/* -*- c++ -*- */

#ifndef PM_H
#define PM_H

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

#if defined(_USE_GPU)

#if defined(_GPU_CUDA)
#include "pm_cuda.h"
#endif

#elif defined(_USE_CPU)

#include "pm_host.h"

#endif


#endif