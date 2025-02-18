#if defined(_GPU_CUBLAS)

#ifndef MATHLIB_CUBLAS_H
#define MATHLIB_CUBLAS_H

#include "../pm/pm.h"

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

namespace MATHLIB_NS {

  class MATHLIB {
    
  public:

    MATHLIB(class PM_NS::PM * pm);
    ~MATHLIB() {};

    int create_handle();
    void set_handle(int);
    void set_handle();
    cublasHandle_t * get_handle();
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
    
    std::vector<cublasHandle_t> my_handles;
    cublasHandle_t * current_handle;
    int current_handle_id;
  };

}

#endif

#endif
