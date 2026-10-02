/* -*- c++ -*- */

#ifndef MATHLIB_H
#define MATHLIB_H

//#define _DEBUG_ML
//#define _PROFILE_ML

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

#if defined(_PROFILE_ML)
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace MATHLIB_NS {

  // Single home for the PROFILE_ML tally and, more importantly, for the log format:
  // every backend builds its name here, so cuda/hip/sycl/host cannot drift apart.
  // The name string is verbatim `-replay` syntax for gemm/gemm_batch -- see
  // mini-apps/math/benchmark (main.cpp parses it, replay_profile.py scrapes it).
  class ProfileML {

  public:

    void record(const std::string & name)
    {
      auto it = std::find(name_.begin(), name_.end(), name);
      size_t indx = it - name_.begin();
      if(indx < name_.size()) count_[indx]++;
      else { name_.push_back(name); count_.push_back(1); }
    }

    void dump() const
    {
      printf("\nLIBGPU :: PROFILE_ML\n");
      for(size_t i=0; i<name_.size(); ++i)
	printf("LIBGPU :: PROFILE_ML :: count= %i  name= %s\n", count_[i], name_[i].c_str());
      fflush(stdout);
    }

    // -- name builders; trans args are read as single chars so a non-NUL-terminated
    // -- char (as the host gemm_batch loop used to pass) cannot run off the end

    static std::string gemm(const char * ta, const char * tb,
			    int m, int n, int k, int lda, int ldb, int ldc,
			    double alpha, double beta)
    {
      std::ostringstream s;
      s << "gemm " << ta[0] << " " << tb[0] << " " << m << " " << n << " " << k
	<< " " << lda << " " << ldb << " " << ldc << " " << alpha << " " << beta;
      return s.str();
    }

    static std::string gemm_batch(const char * ta, const char * tb,
				  int m, int n, int k, int lda, int ldb, int ldc,
				  double alpha, double beta, int batchCount,
				  int strideA, int strideB, int strideC)
    {
      std::ostringstream s;
      s << gemm(ta, tb, m, n, k, lda, ldb, ldc, alpha, beta).substr(5)
	<< " " << batchCount << " " << strideA << " " << strideB << " " << strideC;
      return "gemm_batch " + s.str();
    }

    // gemv mirrors gemm's field order: shape, leading dim + increments, scalars,
    // then (batched) count and strides. alpha is included so the record is a
    // complete, replayable description of the call.
    static std::string gemv(const char * ta, int m, int n, int lda,
			    int incx, int incy, double alpha, double beta)
    {
      std::ostringstream s;
      s << "gemv " << ta[0] << " " << m << " " << n << " " << lda
	<< " " << incx << " " << incy << " " << alpha << " " << beta;
      return s.str();
    }

    static std::string gemv_batch(const char * ta, int m, int n, int lda,
				  int incx, int incy, double alpha, double beta,
				  int batchCount, int strideA, int strideX, int strideY)
    {
      std::ostringstream s;
      s << gemv(ta, m, n, lda, incx, incy, alpha, beta).substr(5)
	<< " " << batchCount << " " << strideA << " " << strideX << " " << strideY;
      return "gemv_batch " + s.str();
    }

  private:

    std::vector<std::string> name_;
    std::vector<int> count_;
  };

}
#endif

#if defined(_USE_GPU)

// set default based on backend if one not explicitly set
#if !defined(_GPU_CUBLAS) && !defined(_GPU_MKL) && !defined(_GPU_HIPBLAS)

#if defined(_GPU_CUDA) || defined(_GPU_SYCL_CUDA)
#error "Did you forget to set -D_GPU_CUBLAS?"
#elif defined(_GPU_SYCL)
#error "Did you forget to set -D_GPU_MKL?"
#elif defined(_GPU_HIP)
#error "Did you forget to set -D_GPU_HIPBLAS?"
#endif

#endif

// load appropriate header

#if defined(_GPU_CUBLAS)
#include "mathlib_cublas.h"
#elif defined(_GPU_MKL)
#include "mathlib_mkl.h"
#elif defined(_GPU_HIPBLAS)
#include "mathlib_hipblas.h"
#endif

#elif defined(_USE_CPU)

#include "mathlib_host.h"

#endif

#endif
