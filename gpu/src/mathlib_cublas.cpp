#if defined(_GPU_CUBLAS)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB(class PM_NS::PM * pm_)
{
  pm = pm_;
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

void MATHLIB::gemm(const char * transa, const char * transb,
		   const int * m, const int * n, const int * k,
		   const double * alpha, const double * a, const int * lda,
		   const double * b, const int * ldb,
		   const double * beta, double * c, const int * ldc)
{

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
}

// ----------------------------------------------------------------

void MATHLIB::gemm_batch(const char * transa, const char * transb,
			 const int * m, const int * n, const int * k,
			 const double * alpha, const double * a, const int * lda, const int * strideA,
			 const double * b, const int * ldb, const int * strideB,
			 const double * beta, double * c, const int * ldc, const int * strideC,
			 const int * batchCount)
{

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
}

#endif
