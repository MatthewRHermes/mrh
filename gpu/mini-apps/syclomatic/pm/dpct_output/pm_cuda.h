#if defined(_GPU_CUDA)

#ifndef PM_CUDA_H
#define PM_CUDA_H

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>

#ifdef _USE_NVTX
#include "nvToolsExt.h"
#endif

#include <iostream>
#include <vector>

namespace PM_NS {

/*
DPCT1001:2: The statement could not be removed.
*/
/*
DPCT1000:3: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1010:4: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
/*
DPCT1009:5: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced by a placeholder string. You need to rewrite this
code.
*/
#define _CUDA_CHECK_ERRORS()                                                   \
  {                                                                            \
    dpct::err0 err = 0;                                                        \
    if (err != 0) {                                                            \
      std::cout << "CUDA error with code "                                     \
                << "<Placeholder string>"                                      \
                << " in file " << __FILE__ << " at line " << __LINE__          \
                << ". Exiting...\n";                                           \
      exit(1);                                                                 \
    }                                                                          \
  }

/*
DPCT1001:14: The statement could not be removed.
*/
/*
DPCT1000:15: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1010:16: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
/*
DPCT1009:17: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced by a placeholder string. You need to rewrite this
code.
*/
#define _CUDA_CHECK_ERRORS2()                                                  \
  {                                                                            \
    dpct::err0 err = 0;                                                        \
    if (err != 0) {                                                            \
      std::cout << "CUDA error with code "                                     \
                << "<Placeholder string>"                                      \
                << " in file " << __FILE__ << " at line " << __LINE__          \
                << ". Exiting...\n";                                           \
      return 1;                                                                \
    }                                                                          \
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

    void* dev_malloc(size_t);
    void* dev_malloc_async(size_t);
    void *dev_malloc_async(size_t, dpct::queue_ptr &s);
    void* dev_malloc_host(size_t);

    void dev_free(void*);
    void dev_free_async(void*);
    void dev_free_async(void *, dpct::queue_ptr &s);
    void dev_free_host(void*);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    void dev_barrier();
    
    int dev_push_async(void * d_ptr, void * h_ptr, size_t N);
    int dev_push_async(void *d_ptr, void *h_ptr, size_t N, dpct::queue_ptr &s);

    void dev_pull_async(void * d_ptr, void * h_ptr, size_t N);
    void dev_pull_async(void *d_ptr, void *h_ptr, size_t N, dpct::queue_ptr &s);

    void dev_enable_peer(int, int);
    void dev_memcpy_peer(void * d_ptr, int dest, void * s_ptr, int src, size_t N);
    void dev_memcpy_peer_async(void * d_ptr, int dest, void * s_ptr, int src, size_t N);
    
    void dev_check_pointer(int, const char *, void *);

    int dev_stream_create();
    void dev_stream_create(dpct::queue_ptr &s);
    void dev_stream_destroy();
    void dev_stream_destroy(dpct::queue_ptr &s);
    void dev_stream_wait();
    void dev_stream_wait(dpct::queue_ptr &s);

    void dev_set_queue(int id);
    dpct::queue_ptr *dev_get_queue();

    void dev_profile_start(const char *);
    void dev_profile_stop();
    void dev_profile_next(const char *);
    
  private:
    void uuid_print(std::array<unsigned char, 16>);

    std::vector<dpct::queue_ptr> my_queues;
    dpct::queue_ptr *current_queue;
    int current_queue_id;
  };

}
#endif

#endif
