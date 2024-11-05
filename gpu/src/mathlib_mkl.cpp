#if defined(_GPU_MKL)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB(class PM_NS::PM * pm)
{
  pm_ = pm;
}

// ----------------------------------------------------------------

void MATHLIB::gemm(const char * transa, const char * transb,
	      const int * m, const int * n, const int * k,
	      const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb,
	      const double * beta, double * c, const int * ldc)
{  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose transA = (transa == "N") ? transpose::nontrans : transpose::trans;
  transpose transB = (transb == "N") ? transpose::nontrans : transpose::trans;
    
#if defined(_GPU_SYCL_CUDA)
  using oneapi::mkl::blas::column_major::gemm;
  
  gemm(*q, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
#else
  using oneapi::mkl::blas::gemm;
  
  gemm(*q, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
#endif
  
}

#endif
