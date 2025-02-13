#if defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)

#ifndef PM_SYCL_H
#define PM_SYCL_H

#include <sycl/sycl.hpp>

#if defined(_GPU_SYCL_CUDA)
#include "cublas_v2.h" // this shouldn't be here. move it to mathlib_cublas.h
#include "nvToolsExt.h"
#endif

#include <iostream>

namespace PM_NS {

#if defined(_GPU_SYCL_CUDA)
#define _SYCL_CHECK_ERRORS()               \
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
  
#define _SYCL_CHECK_ERRORS2()              \
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
  
#else
  
#define _SYCL_CHECK_ERRORS()              \
  {                                        \
  }

#define _SYCL_CHECK_ERRORS2()             \
  {                                        \
  }
#endif
  
  class PM {
    
  public:
    
    PM();
    ~PM() {};
    
    int dev_num_devices();
    void dev_properties(int);

    int dev_check_peer(int, int);
    void dev_check_errors();

    void dev_set_device(int);
    int dev_get_device();

    void* dev_malloc(size_t);
    void* dev_malloc_async(size_t);
    void* dev_malloc_async(size_t, sycl::queue &q);
    void* dev_malloc_host(size_t);

    void dev_free(void*);
    void dev_free_async(void*);
    void dev_free_async(void*, sycl::queue &q);
    void dev_free_host(void*);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    void dev_barrier();
    
    int dev_push_async(void * d_ptr, void * h_ptr, size_t N);
    int dev_push_async(void * d_ptr, void * h_ptr, size_t N, sycl::queue &s);
    
    void dev_pull_async(void * d_ptr, void * h_ptr, size_t N);
    void dev_pull_async(void * d_ptr, void * h_ptr, size_t N, sycl::queue &s);

    void dev_check_pointer(int, const char *, void *);

    int dev_stream_create();
    void dev_stream_destroy();
    void dev_stream_wait();
    
#if defined(_GPU_SYCL_CUDA)
    void dev_stream_create(cudaStream_t & s);
    void dev_stream_destroy(cudaStream_t & s);
    void dev_stream_wait(cudaStream_t & s);
#else
    void dev_stream_create(sycl::queue & q);
    void dev_stream_destroy(sycl::queue & q);
    void dev_stream_wait(sycl::queue & q);
#endif
    
    void dev_set_queue(int);
    sycl::queue * dev_get_queue();
    
  private:
    
    void uuid_print(std::array<unsigned char, 16>);
    
    std::vector<sycl::queue> my_queues;
    sycl::queue * current_queue;
    int current_queue_id;

    int num_devices;
  };

}
#endif

#endif
