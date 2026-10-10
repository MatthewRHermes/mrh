#if defined(_GPU_CUBLAS)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB(class PM_NS::PM * pm_)
{
  pm = pm_;
}

// ----------------------------------------------------------------

MATHLIB::~MATHLIB()
{
#if defined(_PROFILE_ML)
  profile_.dump();
#endif
}

// ----------------------------------------------------------------

int MATHLIB::create_handle()
{
  cublasHandle_t h;

  cublasCreate(&h);
  
  _CUDA_CHECK_ERRORS();
  
  cudaStream_t * s = pm->dev_get_queue();
  
  cublasSetStream(h, *s);
  
  _CUDA_CHECK_ERRORS();

  my_handles.push_back(h);

  int id = my_handles.size() - 1;

  set_handle(id);
  
  return id;
}

// ----------------------------------------------------------------

void MATHLIB::set_handle(int id)
{
  current_handle = &(my_handles[id]);
  current_handle_id = id;
}

// ----------------------------------------------------------------

void MATHLIB::set_handle()
{
  int id = pm->dev_get_device();
  
  current_handle = &(my_handles[id]);
  current_handle_id = id;
}

// ----------------------------------------------------------------

cublasHandle_t * MATHLIB::get_handle()
{  
  return current_handle;
}

// ----------------------------------------------------------------

void MATHLIB::destroy_handle()
{
  int id = current_handle_id;
  
  cublasDestroy(my_handles[id]);
  my_handles[id] = NULL;
}

// ----------------------------------------------------------------

void MATHLIB::memset(double * array, const int * val, const int * size)
{
#ifdef _DEBUG_ML 
  printf("Inside MATHLIB::memset()\n");
#endif
//TODO: add profiling lines related things

#if 1
  cudaStream_t * s = pm->dev_get_queue();

  cudaMemsetAsync ( array, *val, *size, *s);
#else
  cudaMemset ( array, *val, *size);
#endif
  
  _CUDA_CHECK_ERRORS();

#ifdef _DEBUG_ML 
  printf(" -- Leaving MATHLIB::memset()\n");
#endif

}
// ----------------------------------------------------------------

void MATHLIB::memset(double * array, const int * val, const size_t * size)
{
#ifdef _DEBUG_ML 
  printf("Inside MATHLIB::memset()\n");
#endif
//TODO: add profiling lines related things

#if 1
  cudaStream_t * s = pm->dev_get_queue();

  cudaMemsetAsync ( array, *val, *size, *s);
#else
  cudaMemset ( array, *val, *size);
#endif
  
  _CUDA_CHECK_ERRORS();

#ifdef _DEBUG_ML 
  printf(" -- Leaving MATHLIB::memset()\n");
#endif

}


// ----------------------------------------------------------------

void MATHLIB::axpy(const int * n, 
                   const double * alpha, const double * x, const int * incx,
                   double * y, const int * incy)
{
#ifdef _DEBUG_ML 
  printf("Inside MATHLIB::axpy()\n");
#endif
//TODO: add profiling lines related things

  cublasHandle_t * h = current_handle;
  
#ifdef _SINGLE_PRECISION
  cublasSaxpy (*h , *n, alpha, x, *incx, y, *incy);   
#else
  cublasDaxpy (*h , *n, alpha, x, *incx, y, *incy);   
#endif
  
  _CUDA_CHECK_ERRORS();

#ifdef _DEBUG_ML 
  printf(" -- Leaving MATHLIB::axpy()\n");
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
  printf("Inside MATHLIB::gemv()\n");
#endif
  
#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemv_batch(transa, *m, *n, *lda, *incx, *incy, *alpha, *beta,
					*batchCount, *strideA, *strideX, *strideY));
#endif
  
  cublasHandle_t * h = current_handle;
  
  cublasOperation_t ta;
  
  if(strcmp(transa, "N") == 0) ta = CUBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = CUBLAS_OP_T;
  else ta = CUBLAS_OP_C;

#ifdef _SINGLE_PRECISION
  cublasSgemvStridedBatched(*h, ta, *m, *n, 
                            alpha, a, *lda, *strideA, x, *incx, *strideX, beta, y, *incy, *strideY, *batchCount);
#else
  cublasDgemvStridedBatched(*h, ta, *m, *n, 
                            alpha, a, *lda, *strideA, x, *incx, *strideX, beta, y, *incy, *strideY, *batchCount);
#endif
  
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf(" -- Leaving MATHLIB::gemv_batch()\n");
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
  printf("Inside MATHLIB::gemv()\n");
#endif

#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemv(transa, *m, *n, *lda, *incx, *incy, *alpha, *beta));
#endif
 
  cublasHandle_t * h = current_handle;
  
  cublasOperation_t ta;
  
  if(strcmp(transa, "N") == 0) ta = CUBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = CUBLAS_OP_T;
  else ta = CUBLAS_OP_C;

#ifdef _SINGLE_PRECISION
  cublasSgemv(*h, ta, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
#else
  cublasDgemv(*h, ta, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
#endif
  
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf(" -- Leaving MATHLIB::gemv()\n");
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
  printf("Inside MATHLIB::gemm()\n");
#endif

#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemm(transa, transb, *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta));
#endif
  
  cublasHandle_t * h = current_handle;
  
  cublasOperation_t ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = CUBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = CUBLAS_OP_T;
  else ta = CUBLAS_OP_C;

  if(strcmp(transb, "N") == 0) tb = CUBLAS_OP_N;
  else if(strcmp(transb, "T") == 0) tb = CUBLAS_OP_T;
  else tb = CUBLAS_OP_C;
  
#ifdef _SINGLE_PRECISION
  cublasSgemm(*h, ta, tb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#else
  cublasDgemm(*h, ta, tb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#endif
  
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf(" -- Leaving MATHLIB::gemm()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemm_batch(const char * transa, const char * transb,
			 const int * m, const int * n, const int * k,
			 const double * alpha, const double * a, const int * lda, const int * strideA,
			 const double * b, const int * ldb, const int * strideB,
			 const double * beta, double * c, const int * ldc, const int * strideC,
			 const int * batchCount)
{
#ifdef _DEBUG_ML
  printf("Inside MATHLIB::gemm_batch()\n");
  printf("mnk= %i %i %i  alpha= %f  beta= %f  ld= %i %i %i  stride= %i %i %i  batchCount= %i\n",
	 *m,*n,*k,*alpha,*beta,*lda,*ldb,*ldc,*strideA,*strideB,*strideC,*batchCount);
#endif
  
#if defined(_PROFILE_ML)
  profile_.record(ProfileML::gemm_batch(transa, transb, *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta,
					*batchCount, *strideA, *strideB, *strideC));
#endif
  
  cublasHandle_t * h = current_handle;

  cublasOperation_t ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = CUBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = CUBLAS_OP_T;
  else ta = CUBLAS_OP_C;

  if(strcmp(transb, "N") == 0) tb = CUBLAS_OP_N;
  else if(strcmp(transb, "T") == 0) tb = CUBLAS_OP_T;
  else tb = CUBLAS_OP_C;
  
#ifdef _SINGLE_PRECISION
  cublasSgemmStridedBatched(*h, ta, tb, *m, *n, *k,
			    alpha, a, *lda, *strideA, b, *ldb, *strideB, beta, c, *ldc, *strideC, *batchCount);
#else
  cublasDgemmStridedBatched(*h, ta, tb, *m, *n, *k,
			    alpha, a, *lda, *strideA, b, *ldb, *strideB, beta, c, *ldc, *strideC, *batchCount);
#endif

  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf("Leaving MATHLIB::gemm_batch()\n");
#endif
}

#endif
