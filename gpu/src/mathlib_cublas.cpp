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
  
  cudaStream_t * s = pm->dev_get_queue();
  
  cublasSetStream(h, *s);

  my_handles.push_back(h);

  int id = my_handles.size() - 1;
  
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
  
#ifdef _SINGLE_PRECISION
  cublasSgemm(*h, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#else
  cublasDgemm(*h, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#endif
  
}

#endif
