#if defined(_USE_CPU)

#ifndef MATHLIB_HOST_H
#define MATHLIB_HOST_H

#include "pm.h"

namespace MATHLIB_NS {

  class MATHLIB {

  public:

    MATHLIB(class PM_NS::PM * pm);
    ~MATHLIB() {};

    void gemm(const char * transa, const char * transb,
	      const int * m, const int * n, const int * k,
	      const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb,
	      const double * beta, double * c, const int * ldc, void * q);

  private:
    class PM_NS::PM * pm_;
  };

}

#endif

#endif
