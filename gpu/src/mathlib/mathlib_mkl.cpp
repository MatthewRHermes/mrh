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
  std::ostringstream name_;
  name_ << "gemm " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " " << *alpha << " " << *beta;
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
  printf(" -- Leaving MATHLIB::gemm()\n");
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
  printf("Inside MATHLIB::gemm_batch()\n");
  printf("mnk= %i %i %i  alpha= %f  beta= %f  ld= %i %i %i  stride= %i %i %i  batchCount= %i\n",
	 *m,*n,*k,*alpha,*beta,*lda,*ldb,*ldc,*strideA,*strideB,*strideC,*batchCount);
#endif
  
#if defined(_PROFILE_ML)
  std::ostringstream name_;
  name_ << "gemm_batch " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " " << *alpha << " " << *beta << " " << *batchCount;
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
  printf("Leaving MATHLIB::gemm_batch()\n");
#endif
}

#endif
