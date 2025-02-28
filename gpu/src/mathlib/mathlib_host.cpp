#if defined(_USE_CPU)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

extern "C" {
  void dsymm_(const char*, const char*, const int*, const int*,
	      const double*, const double*, const int*,
	      const double*, const int*,
	      const double*, double*, const int*);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
	      const int * k, const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb, const double * beta, double * c,
	      const int * ldc);
}

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
#ifdef _SINGLE_PRECISION
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif 
}

// ----------------------------------------------------------------

void MATHLIB::gemm_batch(const char * transa, const char * transb,
			 const int * m, const int * n, const int * k,
			 const double * alpha, const double * a, const int * lda, const int * strideA,
			 const double * b, const int * ldb, const int * strideB,
			 const double * beta, double * c, const int * ldc, const int * strideC, const int * batchCount)
{
  
#pragma omp parallel for
  for(int i=0; i<*batchCount; ++i) {
    int offset_a = i * (*strideA);
    int offset_b = i * (*strideB);
    int offset_c = i * (*strideC);
    
    const real_t * a_ = &(a[offset_a]);
    const real_t * b_ = &(b[offset_b]);
    real_t * c_ = &(c[offset_c]);
    
#ifdef _SINGLE_PRECISION
    sgemm_(transa, transb, m, n, k, alpha, a_, lda, b_, ldb, beta, c_, ldc);
#else
    dgemm_(transa, transb, m, n, k, alpha, a_, lda, b_, ldb, beta, c_, ldc);
#endif
  }
  
}

#endif
