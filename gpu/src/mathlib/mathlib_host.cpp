#if defined(_USE_CPU)

#include <string.h>

// via mathlib.h (not mathlib_host.h) so this TU sees the same _PROFILE_ML
// setting as everything else -- otherwise MATHLIB has a different layout here
#include "mathlib.h"

using namespace MATHLIB_NS;

extern "C" {
  void dgemv_(const char * trans, const int * m, const int * n,
              const double * alpha, const double * a, const int * lda,
              const double * x, const int * incx,
              const double * beta, double * y, const int * incy);
  void daxpy_(const int * n, const double * alpha, const double * x, const int * incx,
              double * y, const int * incy);
  void dgemm_(const char * transa, const char * transb,
              const int * m, const int * n, const int * k,
              const double * alpha, const double * a, const int * lda,
              const double * b, const int * ldb,
              const double * beta, double * c, const int * ldc);
  void dsymm_(const char * side, const char * uplo,
              const int * m, const int * n,
              const double * alpha, const double * a, const int * lda,
              const double * b, const int * ldb,
              const double * beta, double * c, const int * ldc);
}

#if defined(_SINGLE_PRECISION)
extern "C" {
  void sgemv_(const char * trans, const int * m, const int * n,
              const float * alpha, const float * a, const int * lda,
              const float * x, const int * incx,
              const float * beta, float * y, const int * incy);
  void saxpy_(const int * n, const float * alpha, const float * x, const int * incx,
              float * y, const int * incy);
  void sgemm_(const char * transa, const char * transb,
              const int * m, const int * n, const int * k,
              const float * alpha, const float * a, const int * lda,
              const float * b, const int * ldb,
              const float * beta, float * c, const int * ldc);
  void ssymm_(const char * side, const char * uplo,
              const int * m, const int * n,
              const float * alpha, const float * a, const int * lda,
              const float * b, const int * ldb,
              const float * beta, float * c, const int * ldc);
}
#endif



MATHLIB::~MATHLIB()
{
#if defined(_PROFILE_ML)
  profile_.dump();
#endif
}

MATHLIB::MATHLIB(class PM_NS::PM * pm) : pm_(pm)
{
}

void MATHLIB::memset(double * array, const int * val, const int * size)
{
  ::memset(array, *val, (size_t)(*size));
}

void MATHLIB::memset(double * array, const int * val, const size_t * size)
{
  ::memset(array, *val, *size);
}

void MATHLIB::axpy(const int * n, const double * alpha, const double * x, const int * incx,
                   double * y, const int * incy)
{
#if defined(_SINGLE_PRECISION)
  saxpy_(n, alpha, x, incx, y, incy);
#else
  daxpy_(n, alpha, x, incx, y, incy);
#endif
}

void MATHLIB::gemv(const char * transa, const int * m, const int * n,
                   const double * alpha, const double * a, const int * lda,
                   const double * x, const int * incx,
                   const double * beta, double * y, const int * incy)
{
#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemv(transa, *m, *n, *lda, *incx, *incy, *alpha, *beta));
#endif

#if defined(_SINGLE_PRECISION)
  sgemv_(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
#else
  dgemv_(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}

void MATHLIB::gemv_batch(const char * transa, const int * m, const int * n,
                         const double * alpha, const double * a, const int * lda, const int * strideA,
                         const double * x, const int * incx, const int * strideX,
                         const double * beta, double * y, const int * incy, const int * strideY,
                         const int * batchCount)
{
#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemv_batch(transa, *m, *n, *lda, *incx, *incy, *alpha, *beta,
					*batchCount, *strideA, *strideX, *strideY));
#endif

  const char trans = transa[0];
  for (int b = 0; b < *batchCount; ++b) {
    const double * ab = a + b * (*strideA);
    const double * xb = x + b * (*strideX);
    double * yb = y + b * (*strideY);
    gemv(&trans, m, n, alpha, ab, lda, xb, incx, beta, yb, incy);
  }
}

void MATHLIB::gemm(const char * transa, const char * transb,
                   const int * m, const int * n, const int * k,
                   const double * alpha, const double * a, const int * lda,
                   const double * b, const int * ldb,
                   const double * beta, double * c, const int * ldc)
{
#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemm(transa, transb, *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta));
#endif

  gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void MATHLIB::gemm_impl(const char * transa, const char * transb,
                   const int * m, const int * n, const int * k,
                   const double * alpha, const double * a, const int * lda,
                   const double * b, const int * ldb,
                   const double * beta, double * c, const int * ldc)
{
  // TEMPORARY DIAGNOSTIC for the k=-1 bug; remove after fix
  static int _last_m[8], _last_n[8], _last_k[8], _last_idx = 0;
  static char _last_ta[8], _last_tb[8];
  if (*m < 0 || *n < 0 || *k < 0) {
    fprintf(stderr, "MATHLIB::gemm BAD DIMS: transa=%c transb=%c m=%d n=%d k=%d lda=%d ldb=%d ldc=%d a=%p b=%p c=%p\n",
            transa[0], transb[0], *m, *n, *k, *lda, *ldb, *ldc, (const void*)a, (const void*)b, (void*)c);
    for (int i = 0; i < 8; ++i) {
      int j = (_last_idx + i) % 8;
      fprintf(stderr, "  last %d: %c%c m=%d n=%d k=%d\n", i, _last_ta[j], _last_tb[j], _last_m[j], _last_n[j], _last_k[j]);
    }
    abort();
  }
  _last_ta[_last_idx] = transa[0]; _last_tb[_last_idx] = transb[0];
  _last_m[_last_idx] = *m; _last_n[_last_idx] = *n; _last_k[_last_idx] = *k;
  _last_idx = (_last_idx + 1) % 8;
#if defined(_SINGLE_PRECISION)
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}

void MATHLIB::gemm_batch(const char * transa, const char * transb,
                         const int * m, const int * n, const int * k,
                         const double * alpha, const double * a, const int * lda, const int * strideA,
                         const double * b, const int * ldb, const int * strideB,
                         const double * beta, double * c, const int * ldc, const int * strideC,
                         const int * batchCount)
{
#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemm_batch(transa, transb, *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta,
					*batchCount, *strideA, *strideB, *strideC));
#endif

  // NUL-terminate: these are passed as const char*, and anything that treats them
  // as a C string (the PROFILE_ML name, for one) runs off the end of a bare char
  const char tA[2] = { transa[0], '\0' };
  const char tB[2] = { transb[0], '\0' };
  for (int i = 0; i < *batchCount; ++i) {
    const double * ai = a + i * (*strideA);
    const double * bi = b + i * (*strideB);
    double * ci = c + i * (*strideC);
    gemm_impl(tA, tB, m, n, k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
  }
}

#endif
