#if defined(_GPU_OPENMP)

#ifndef PM_OPENMP_H
#define PM_OPENMP_H

#include <omp.h>

// this is needed for math libraries; would be replaced with MKL on Intel
#include "cublas_v2.h"

// this is specific to NVIDIA tools; could have something similar with Intel
#include "nvToolsExt.h"

namespace PM_NS {
  
#define _OMP_CHECK_ERRORS()                \
  {					   \
    					   \
  }

  class PM {

  public:

    PM();
    ~PM() {};
  
    int dev_num_devices();
    void dev_properties(int);
    int dev_check_peer(int, int);

    void dev_set_device(int);
    //    int dev_get_device();

    void* dev_malloc(int);
    void* dev_malloc_host(int);

    void dev_free(void*);
    void dev_free_host(void*);

    void dev_push(void*, void*, int);
    void dev_pull(void*, void*, int);
    void dev_copy(void*, void*, int);

    //void dev_check_pointer(int, const char *, void *);

  };

}

#endif

#endif
