#if defined(_GPU_CUDA)

#ifndef PM_CUDA_H
#define PM_CUDA_H

#include <iostream>

#define MIN(A,B) ((A) < (B) ? (A) : (B))

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

extern void dev_check_pointer(int, const char *, void *);

#endif

#endif
