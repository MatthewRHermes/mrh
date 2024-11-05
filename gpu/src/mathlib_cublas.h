#if defined(_GPU_CUBLAS)

#ifndef MATHLIB_CUBLAS_H
#define MATHLIB_CUBLAS_H

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

namespace MATHLIB_NS {

  class MATHLIB {

  public:

    MATHLIB();
    ~MATHLIB() {};

    void gemm(const char * transa, const char * transb,
	      const int * m, const int * n, const int * k,
	      const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb,
	      const double * beta, double * c, const int * ldc, cublasHandle_t & q);
  };

}

#endif

#endif
