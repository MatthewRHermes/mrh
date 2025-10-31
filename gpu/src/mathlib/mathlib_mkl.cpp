#if defined(_GPU_MKL)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB(class PM_NS::PM * pm)
{
  pm_ = pm;
}

// ----------------------------------------------------------------

MATHLIB::~MATHLIB()
{
#if defined(_PROFILE_ML)
  printf("\nLIBGPU :: PROFILE_ML\n");
  for(int i=0; i<profile_name.size(); ++i) {
    printf("LIBGPU :: PROFILE_ML :: count= %i  name= %s\n", profile_count[i], profile_name[i].c_str());
  }
#endif
}

// ----------------------------------------------------------------

void MATHLIB::memset(double * array, const int * num, const int * size)
{
#ifdef _DEBUG_ML 
  printf("LIBGPU :: Inside MATHLIB::memset()\n");
#endif

  sycl::queue * q = pm_->dev_get_queue();
  
#if 1
  q->memset(array, *num, *size);
#else
  q->memset(array, *num, *size).wait();
#endif

#ifdef _DEBUG_ML
  pm_->dev_stream_wait();
  printf("LIBGPU ::  -- Leaving MATHLIB::memset()\n");
#endif

}

// ----------------------------------------------------------------

void MATHLIB::axpy(const int * n, 
                   const double * alpha, const double * x, const int * incx,
                   double * y, const int * incy)
{
#ifdef _DEBUG_ML 
  printf("LIBGPU :: Inside MATHLIB::axpy()\n");
#endif

  sycl::queue * q = pm_->dev_get_queue();

#if defined(_GPU_SYCL_CUDA)
  oneapi::mkl::blas::column_major::axpy(*q, *n, *alpha, x, *incx, y, *incy);
#else
  oneapi::mkl::blas::axpy(*q, *n, *alpha, x, *incx, y, *incy);  
#endif

#ifdef _DEBUG_ML
  pm_->dev_stream_wait();
  printf("LIBGPU ::  -- Leaving MATHLIB::axpy()\n");
#endif

}

// ----------------------------------------------------------------

void MATHLIB::gemv_batch(const char * transa,
		   const int * m, const int * n, 
		   const double * alpha, const double * a, const int * lda, const int * strideA,
		   const double * x, const int * incx, const int * strideX,
		   const double * beta, double * y, const int * incy, const int * strideY, 
                   const int * batchCount)
{
#ifdef _DEBUG_ML
  printf("LIBGPU :: Inside MATHLIB::gemv_batch()\n");
  printf("LIBGPU :: mn= %i %i  alpha= %f  beta= %f  ld= %i inc= %i %i  stride= %i %i %i  batchCount= %i\n",
	 *m,*n,*alpha,*beta,*lda,*incx,*incy,*strideA,*strideX,*strideY,*batchCount);
#endif
  
#if defined(_PROFILE_ML)
  std::ostringstream name_;
  name_ << "gemv_batch " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " "
	<< *lda << " " << *ldb << " " << *ldc << " " << *alpha << " " << *beta << " " << *batchCount;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif
  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose ta;
  
  if(strcmp(transa, "N") == 0) ta = transpose::nontrans;
  else if(strcmp(transa, "T") == 0) ta = transpose::trans;
  else ta = transpose::conjtrans;
  
#if defined(_GPU_SYCL_CUDA)  
  oneapi::mkl::blas::column_major::gemv_batch(*q, ta, *m, *n, *alpha,
					      a, *lda, *strideA, x, *incx, *strideX, *beta, y, *incy, *strideY, *batchCount);
#else
  oneapi::mkl::blas::gemv_batch(*q, ta, *m, *n, *alpha,
					      a, *lda, *strideA, x, *incx, *strideX, *beta, y, *incy, *strideY, *batchCount);
#endif
  
#ifdef _DEBUG_ML
  pm_->dev_stream_wait();
  printf("LIBGPU ::  -- Leaving MATHLIB::gemv_batch()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemv(const char * transa,
		   const int * m, const int * n, 
		   const double * alpha, const double * a, const int * lda,
		   const double * x, const int * incx,
		   const double * beta, double * y, const int * incy)
{
#ifdef _DEBUG_ML
  printf("LIBGPU :: Inside MATHLIB::gemv()\n");
#endif

//#if defined(_PROFILE_ML)
#if 0
  std::ostringstream name_;
  name_ << "gemv " << transa << " "  << *m << " " << *n << " "
	<< *lda << " " << *ldb << " " << *ldc << " " << *alpha << " " << *beta;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif
  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = transpose::nontrans;
  else if(strcmp(transa, "T") == 0) ta = transpose::trans;
  else ta = transpose::conjtrans;

#if defined(_GPU_SYCL_CUDA)  
  oneapi::mkl::blas::column_major::gemv(*q, ta, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
#else
  oneapi::mkl::blas::gemv(*q, ta, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
#endif

#ifdef _DEBUG_ML
  printf("LIBGPU ::  -- Leaving MATHLIB::gemv()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemm(const char * transa, const char * transb,
		   const int * m, const int * n, const int * k,
		   const double * alpha, const double * a, const int * lda,
		   const double * b, const int * ldb,
		   const double * beta, double * c, const int * ldc)
{  
#ifdef _DEBUG_ML
  printf("LIBGPU :: Inside MATHLIB::gemm()\n");
#endif

#if defined(_PROFILE_ML)
  std::ostringstream name_;
  name_ << "gemm " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " "
	<< *lda << " " << *ldb << " " << *ldc << " " << *alpha << " " << *beta;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif
  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = transpose::nontrans;
  else if(strcmp(transa, "T") == 0) ta = transpose::trans;
  else ta = transpose::conjtrans;
  
  if(strcmp(transb, "N") == 0) tb = transpose::nontrans;
  else if(strcmp(transb, "T") == 0) tb = transpose::trans;
  else tb = transpose::conjtrans;
    
#if defined(_GPU_SYCL_CUDA)
  oneapi::mkl::blas::column_major::gemm(*q, ta, tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
#else
  oneapi::mkl::blas::gemm(*q, ta, tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
#endif
  
#ifdef _DEBUG_ML
  printf("LIBGPU ::  -- Leaving MATHLIB::gemm()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemm_batch(const char * transa, const char * transb,
			 const int * m, const int * n, const int * k,
			 const double * alpha, const double * a, const int * lda, const int * strideA,
			 const double * b, const int * ldb, const int * strideB,
			 const double * beta, double * c, const int * ldc, const int * strideC, const int * batchCount)
{  
#ifdef _DEBUG_ML
  printf("LIBGPU :: Inside MATHLIB::gemm_batch()\n");
  printf("LIBGPU :: mnk= %i %i %i  alpha= %f  beta= %f  ld= %i %i %i  stride= %i %i %i  batchCount= %i\n",
	 *m,*n,*k,*alpha,*beta,*lda,*ldb,*ldc,*strideA,*strideB,*strideC,*batchCount);
#endif
  
#if defined(_PROFILE_ML)
  std::ostringstream name_;
  name_ << "gemm_batch " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " "
	<< *lda << " " << *ldb << " " << *ldc << " " << *alpha << " " << *beta << " " << *batchCount;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif
  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = transpose::nontrans;
  else if(strcmp(transa, "T") == 0) ta = transpose::trans;
  else ta = transpose::conjtrans;
  
  if(strcmp(transb, "N") == 0) tb = transpose::nontrans;
  else if(strcmp(transb, "T") == 0) tb = transpose::trans;
  else tb = transpose::conjtrans;
    
#if defined(_GPU_SYCL_CUDA)  
  oneapi::mkl::blas::column_major::gemm_batch(*q, ta, tb, *m, *n, *k, *alpha,
					      a, *lda, *strideA, b, *ldb, *strideB, *beta, c, *ldc, *strideC, *batchCount);
#else
  oneapi::mkl::blas::gemm_batch(*q, ta, tb, *m, *n, *k, *alpha,
				a, *lda, *strideA, b, *ldb, *strideB, *beta, c, *ldc, *strideC, *batchCount);
#endif

#ifdef _DEBUG_ML
  printf("LIBGPU :: Leaving MATHLIB::gemm_batch()\n");
#endif
}

#endif
