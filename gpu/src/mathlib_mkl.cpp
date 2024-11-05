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
	      const double * beta, double * c, const int * ldc, void * q_)
{  
  sycl::queue * q = pm_->dev_get_queue();
  
  using oneapi::mkl::blas::column_major::gemm;
  using oneapi::mkl::transpose;
#ifdef _SINGLE_PRECISION
  //  cublasSgemm(q, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#else
  gemm(*q, transpose::nontrans, transpose::nontrans, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
  //  cublasDgemm(q, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
  
#endif
  
}

#endif
