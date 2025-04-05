#if defined(_GPU_MKL)

#ifndef MATHLIB_MKL_H
#define MATHLIB_MKL_H

#include "../pm/pm.h"

#if defined(_GPU_SYCL_CUDA)
#include "oneapi/mkl.hpp"
#else
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#endif

namespace MATHLIB_NS {

  class MATHLIB {

  public:

    MATHLIB(class PM_NS::PM * pm);
    ~MATHLIB() {};

    int create_handle() {return 0;};
    void set_handle(int) {};
    void set_handle() {};
    int * get_handle() {return nullptr;};
    void destroy_handle() {};

    void gemm(const char * transa, const char * transb,
	      const int * m, const int * n, const int * k,
	      const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb,
	      const double * beta, double * c, const int * ldc);
    
    void gemm_batch(const char * transa, const char * transb,
		    const int * m, const int * n, const int * k,
		    const double * alpha, const double * a, const int * lda, const int * strideA,
		    const double * b, const int * ldb, const int * strideB,
		    const double * beta, double * c, const int * ldc, const int * strideC,
		    const int * batchCount);

  private:
    class PM_NS::PM * pm_;
  };

}

#endif

#endif
