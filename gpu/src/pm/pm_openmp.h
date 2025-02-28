#if defined(_GPU_OPENMP)

#ifndef PM_OPENMP_H
#define PM_OPENMP_H

//#include <omp.h>

// this is needed for _CUDA_CHECK_ERRORS()
#include <cuda_runtime_api.h>
#include <iostream>

// this is needed for math libraries; would be replaced with MKL on Intel
#include "cublas_v2.h"

// this is specific to NVIDIA tools; could have something similar with Intel
#include "nvToolsExt.h"

namespace PM_NS {
  
#define _OMP_CHECK_ERRORS(err)             \
  {					   \
    if(err) {		                   \
      std::cout				   \
	<< "OpenMPTarget error with code " \
	<< " in file " << __FILE__	   \
	<< " at line " << __LINE__	   \
	<< ". Exiting...\n";		   \
      exit(1);				   \
    }					   \
  }

#define _CUDA_CHECK_ERRORS()               \
  {                                        \
    cudaError err = cudaGetLastError();    \
    if(err != cudaSuccess) {               \
      std::cout                            \
        << "CUDA error with code "         \
        << cudaGetErrorString(err)         \
        << " in file " << __FILE__         \
        << " at line " << __LINE__         \
        << ". Exiting...\n";               \
      exit(1);                             \
    }                                      \
  }

  class PM {

  public:

    PM();
    ~PM() {};
  
    int dev_num_devices();
    void dev_properties(int);
    int dev_check_peer(int, int);

    void dev_set_device(int);
    int dev_get_device();

    void* dev_malloc(int);
    void* dev_malloc_host(int);

    void dev_free(void*);
    void dev_free_host(void*);

    void dev_push(void*, void*, int);
    void dev_pull(void*, void*, int);
    void dev_copy(void*, void*, int);

    void dev_check_pointer(int, const char *, void *);

    void dev_barrier();
    
    // specific to OpenMP

    void dev_push_async(void*, void*, int, void*);
    void dev_pull_async(void*, void*, int, void*);

    void dev_stream_create(cudaStream_t & s);
    void dev_stream_destroy(cudaStream_t & s);
    void dev_stream_wait(cudaStream_t & s);
  };

}

#endif

#endif
