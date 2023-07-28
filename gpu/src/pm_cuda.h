#if defined(_GPU_CUDA)

#ifndef PM_CUDA_H
#define PM_CUDA_H

#include "cublas_v2.h"

#include "nvToolsExt.h"

#include <iostream>

#define _CUDA_CHECK_ERRORS()               \
{                                          \
  cudaError err = cudaGetLastError();	   \
  if(err != cudaSuccess) {		   \
    std::cout				   \
      << "CUDA error with code "           \
      << cudaGetErrorString(err)	   \
      << " in file " << __FILE__           \
      << " at line " << __LINE__	   \
      << ". Exiting...\n";		   \
    exit(1);				   \
  }                                        \
}

extern int dev_num_devices();
extern void dev_properties(int);
extern int dev_check_peer(int, int);

extern void dev_set_device(int);
extern int dev_get_device();

extern void* dev_malloc(size_t);
extern void* dev_malloc_host(size_t);

extern void dev_free(void*);
extern void dev_free_host(void*);

extern void dev_push(void*, void*, size_t);
extern void dev_pull(void*, void*, size_t);
extern void dev_copy(void*, void*, size_t);

extern void dev_push_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s);
extern void dev_pull_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s);

extern void dev_check_pointer(int, const char *, void *);

extern void dev_stream_create(cudaStream_t & s);
extern void dev_stream_destroy(cudaStream_t & s);
extern void dev_stream_wait(cudaStream_t & s);
#endif

#endif
