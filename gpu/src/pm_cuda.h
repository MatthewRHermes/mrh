#if defined(_GPU_CUDA)

#ifndef PM_CUDA_H
#define PM_CUDA_H

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#include "nvToolsExt.h"

#include <iostream>

namespace PM_NS {
  
#define _CUDA_CHECK_ERRORS()               \
  {					   \
    cudaError err = cudaGetLastError();	   \
    if(err != cudaSuccess) {		   \
      std::cout				   \
	<< "CUDA error with code "	   \
	<< cudaGetErrorString(err)	   \
	<< " in file " << __FILE__	   \
	<< " at line " << __LINE__	   \
	<< ". Exiting...\n";		   \
      exit(1);				   \
    }					   \
  }
  
#define _CUDA_CHECK_ERRORS2()               \
  {					   \
    cudaError err = cudaGetLastError();	   \
    if(err != cudaSuccess) {		   \
      std::cout				   \
	<< "CUDA error with code "	   \
	<< cudaGetErrorString(err)	   \
	<< " in file " << __FILE__	   \
	<< " at line " << __LINE__	   \
	<< ". Exiting...\n";		   \
      return 1;				   \
    }					   \
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

    void* dev_malloc(size_t);
    void* dev_malloc_host(size_t);

    void dev_free(void*);
    void dev_free_host(void*);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    // specific to cuda
    
    int dev_push_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s);
    void dev_pull_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s);

    void dev_check_pointer(int, const char *, void *);

    void dev_stream_create(cudaStream_t & s);
    void dev_stream_destroy(cudaStream_t & s);
    void dev_stream_wait(cudaStream_t & s);

    void uuid_print(cudaUUID_t);
  };

}
#endif

#endif
