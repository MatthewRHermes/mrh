#if defined(_GPU_CUDA)

#ifndef PM_CUDA_H
#define PM_CUDA_H

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#ifdef _USE_NVTX
#include "nvToolsExt.h"
#endif

#include <iostream>
#include <vector>

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
  
#define _CUDA_CHECK_ERRORS2()              \
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
    ~PM();
    
    int dev_num_devices();
    void dev_properties(int);
    int dev_check_peer(int, int);

    void dev_set_device(int);
    int dev_get_device();

    void* dev_malloc(size_t, std::string, const char *, int);
    void* dev_malloc_async(size_t, std::string, const char *, int);
    void* dev_malloc_async(size_t, cudaStream_t &s, std::string, const char *, int);
    void* dev_malloc_host(size_t);

    void dev_free(void*, std::string = "DEFAULT");
    void dev_free_async(void*, std::string = "DEFAULT");
    void dev_free_async(void*, cudaStream_t &s, std::string = "DEFAULT");
    void dev_free_host(void*);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    void dev_barrier();
    
    int dev_push_async(void * d_ptr, void * h_ptr, size_t N);
    int dev_push_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s);
    
    void dev_pull_async(void * d_ptr, void * h_ptr, size_t N);
    void dev_pull_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s);

    void dev_enable_peer(int, int);
    void dev_memcpy_peer(void * d_ptr, int dest, void * s_ptr, int src, size_t N);
    void dev_memcpy_peer_async(void * d_ptr, int dest, void * s_ptr, int src, size_t N);
    
    void dev_check_pointer(int, const char *, void *);

    int dev_stream_create();
    void dev_stream_create(cudaStream_t & s);
    void dev_stream_destroy();
    void dev_stream_destroy(cudaStream_t & s);
    void dev_stream_wait();
    void dev_stream_wait(cudaStream_t & s);

    void dev_set_queue(int id);
    cudaStream_t * dev_get_queue();
    
    void dev_profile_start(const char *);
    void dev_profile_stop();
    void dev_profile_next(const char *);
    
    void print_mem_summary();
    
  private:
    
    void uuid_print(cudaUUID_t);

    void profile_memory(size_t, std::string, int mode);
    
    std::vector<cudaStream_t> my_queues;
    cudaStream_t * current_queue;
    int current_queue_id;
    
#if defined (_PROFILE_PM_MEM)
    std::vector<std::string> profile_mem_name;
    std::vector<size_t> profile_mem_size;
    std::vector<size_t> profile_mem_max_size;
    std::vector<size_t> profile_mem_count_alloc;
    std::vector<size_t> profile_mem_count_free;
#endif
  };

}
#endif

#endif
