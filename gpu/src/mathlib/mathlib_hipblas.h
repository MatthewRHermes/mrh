#if defined(_GPU_HIPBLAS)

#ifndef MATHLIB_HIPBLAS_H
#define MATHLIB_HIPBLAS_H

#include "../pm/pm.h"

#include "hipblas.h"

namespace MATHLIB_NS {

  class MATHLIB {
    
  public:

    MATHLIB(class PM_NS::PM * pm);
    ~MATHLIB() {};

    int create_handle();
    void set_handle(int);
    void set_handle();
    hipblasHandle_t * get_handle();
    void destroy_handle();
    
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
    class PM_NS::PM * pm;
    
    std::vector<hipblasHandle_t> my_handles;
    hipblasHandle_t * current_handle;
    int current_handle_id;
  };

}

#endif

#endif
