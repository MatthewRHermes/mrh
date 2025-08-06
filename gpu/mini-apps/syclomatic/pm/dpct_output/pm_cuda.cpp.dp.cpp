#if defined(_GPU_CUDA)

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <iostream>

#include <iomanip>
#include <vector>
#include <tuple>

#include "pm.h"

//#define _DEBUG_PM

using namespace PM_NS;

PM::PM()
{
  int num_devices = dev_num_devices();

  // initialize main queue/stream for each device
  
  for(int i=0; i<num_devices; ++i) {
    /*
    DPCT1093:1: The "i" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    dpct::select_device(i);
    _CUDA_CHECK_ERRORS();

    dpct::queue_ptr s;
    s = dpct::get_current_device().create_queue();

    my_queues.push_back(s);
  }

  /*
  DPCT1093:0: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  dpct::select_device(0);
}

PM::~PM()
{
  int n = my_queues.size();
  for (int i = 0; i < n; ++i) dpct::get_current_device().destroy_queue(
      my_queues[i]);
}

//https://stackoverflow.com/questions/68823023/set-cuda-device-by-uuid
void PM::uuid_print(std::array<unsigned char, 16> a) {
  std::cout << "GPU";
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  for (auto t : r){
    std::cout << "-";
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << std::setfill('0') << std::setw(2)
                << (unsigned)(unsigned char)a[i];
  }
  std::cout << std::endl;
}

int PM::dev_num_devices()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_num_devices()\n");
#endif
  int num_devices;

  num_devices = dpct::dev_mgr::instance().device_count();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_num_devices() : num_devices= %i\n",num_devices);
#endif
  
  return num_devices;
}

void PM::dev_properties(int ndev)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_properties()\n");
#endif
  
  for(int i=0; i<ndev; ++i) {
    dpct::device_info prop;
    dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(i));
    _CUDA_CHECK_ERRORS();

    char name[256];
    strcpy(name, prop.get_name());

    printf("LIBGPU ::  [%i] Platform[ Nvidia ] Type[ GPU ] Device[ %s ]  uuid= ", i, name);
    uuid_print(prop.get_uuid());
    printf("\n");
  }

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_properties()\n");
#endif
}

int PM::dev_check_peer(int rank, int ngpus)
{
#ifdef _DEBUG_PM
  if(rank == 0) {
    printf("Inside PM::dev_check_peer()\n");
    printf("\nLIBGPU: Checking P2P Access for ngpus= %i\n",ngpus);
  }
#endif
  
  int err = 0;  
  for(int ig=0; ig<ngpus; ++ig) {
    /*
    DPCT1093:6: The "ig" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    dpct::select_device(ig);
#ifdef _DEBUG_PM
    if(rank == 0) printf("LIBGPU: -- Device i= %i\n",ig);
#endif

    int n = 1;
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) {
        int access;
        access =
            dpct::dev_mgr::instance().get_device(ig).ext_oneapi_can_access_peer(
                dpct::dev_mgr::instance().get_device(jg));
        n += access;
#ifdef _DEBUG_PM	
        if(rank == 0) printf("LIBGPU: --  --  Device j= %i  access= %i\n",jg,access);
#endif
      }
    }
    if(n != ngpus) err += 1;
  }
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_peer()\n");
#endif
  
  return err;
}

void PM::dev_enable_peer(int rank, int ngpus)
{
#ifdef _DEBUG_PM
  if(rank == 0) {
    printf("Inside PM::dev_enable_peer()\n");
    printf("LIBGPU: -- Enabling peer access for ngpus= %i\n",ngpus);
  }
#endif

  for(int ig=0; ig<ngpus; ++ig) {
    /*
    DPCT1093:7: The "ig" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    dpct::select_device(ig);

    for(int jg=0; jg<ngpus; ++jg) {
      if (jg != ig) dpct::get_current_device().ext_oneapi_enable_peer_access(
          dpct::dev_mgr::instance().get_device(jg));
    }
    
  }
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_enable_peer()\n");
#endif
}

void PM::dev_set_device(int id)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_set_device()\n");
#endif

  /*
  DPCT1093:8: The "id" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  dpct::select_device(id);
  _CUDA_CHECK_ERRORS();

  current_queue = &(my_queues[id]);
  current_queue_id = id;

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_set_devices()\n");
#endif
}

int PM::dev_get_device()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_device()\n");
#endif
  
  int id;
  id = dpct::dev_mgr::instance().current_device_id();
  _CUDA_CHECK_ERRORS();

  current_queue_id = id; // matches whatever cudaGetDevice() returns

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_get_device() : id= %i\n",id);
#endif
  
  return id;
}

void * PM::dev_malloc(size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc()\n");
#endif
  
  void * ptr;
  ptr = (void *)sycl::malloc_device(N, dpct::get_in_order_queue());
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc()\n");
#endif
  
  return ptr;
}

void * PM::dev_malloc_async(size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif

  void * ptr;
#ifdef _NO_CUDA_ASYNC
  cudaMalloc((void**) &ptr, N);
#else
  /*
  DPCT1007:9: Migration of cudaMallocAsync is not supported.
  */
  cudaMallocAsync((void **)&ptr, N, *current_queue);
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

void *PM::dev_malloc_async(size_t N, dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif
  
  void * ptr;
#ifdef _NO_CUDA_ASYNC
  cudaMalloc((void**) &ptr, N);
#else
  /*
  DPCT1007:10: Migration of cudaMallocAsync is not supported.
  */
  cudaMallocAsync((void **)&ptr, N, s);
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

void * PM::dev_malloc_host(size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_host()\n");
#endif
  
  void * ptr;
  ptr = (void *)sycl::malloc_host(N, dpct::get_in_order_queue());
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_host()\n");
#endif
  
  return ptr;
}

void PM::dev_free(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free()\n");
#endif

  if (ptr) dpct::dpct_free(ptr, dpct::get_in_order_queue());
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free()\n");
#endif
}

void PM::dev_free_async(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
 
#ifdef _NO_CUDA_ASYNC
  if(ptr) cudaFree(ptr);
#else
  /*
  DPCT1007:11: Migration of cudaFreeAsync is not supported.
  */
  if (ptr) cudaFreeAsync(ptr, *current_queue);
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_async(void *ptr, dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
 
#ifdef _NO_CUDA_ASYNC
  if(ptr) cudaFree(ptr);
#else
  /*
  DPCT1007:12: Migration of cudaFreeAsync is not supported.
  */
  if (ptr) cudaFreeAsync(ptr, s);
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_host(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_host()\n");
#endif

  if (ptr) sycl::free(ptr, dpct::get_in_order_queue());
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_host()\n");
#endif
}

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push()\n");
#endif

  dpct::get_in_order_queue().memcpy(d_ptr, h_ptr, N).wait();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push()\n");
#endif
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif

  /*
  DPCT1124:13: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  *current_queue->memcpy(d_ptr, h_ptr, N);
  _CUDA_CHECK_ERRORS2();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

int PM::dev_push_async(void *d_ptr, void *h_ptr, size_t N, dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif

  /*
  DPCT1124:18: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  s->memcpy(d_ptr, h_ptr, N);
  _CUDA_CHECK_ERRORS2();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

void PM::dev_pull(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull()\n");
#endif

  dpct::get_in_order_queue().memcpy(h_ptr, d_ptr, N).wait();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull()\n");
#endif
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif

  /*
  DPCT1124:19: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  *current_queue->memcpy(h_ptr, d_ptr, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

void PM::dev_pull_async(void *d_ptr, void *h_ptr, size_t N, dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif

  /*
  DPCT1124:20: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  s->memcpy(h_ptr, d_ptr, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

void PM::dev_memcpy_peer(void * d_ptr, int dest, void * s_ptr, int src, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_memcpy_peer()\n");
#endif

  /*
  DPCT1007:21: Migration of cudaMemcpyPeer is not supported.
  */
  cudaMemcpyPeer(d_ptr, dest, s_ptr, src, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_memcpy_peer()\n");
#endif
}

void PM::dev_memcpy_peer_async(void * d_ptr, int dest, void * s_ptr, int src, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_memcpy_peer_async()\n");
#endif

  /*
  DPCT1007:22: Migration of cudaMemcpyPeerAsync is not supported.
  */
  cudaMemcpyPeerAsync(d_ptr, dest, s_ptr, src, N, *current_queue);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_memcpy_peer_async()\n");
#endif
}

void PM::dev_copy(void * dest, void * src, size_t N)
{ 
#ifdef _DEBUG_PM
  printf("Inside PM::dev_copy()\n");
#endif

  dpct::get_in_order_queue().memcpy(dest, src, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_copy()\n");
#endif
}

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_check_pointer()\n");
#endif

  dpct::pointer_attributes attributes;
  attributes.init(ptr);
  if (attributes.get_device_pointer() != NULL)
      printf("(%i) ptr %s is devicePointer\n", rnk, name);
  if (attributes.get_host_pointer() != NULL)
      printf("(%i) ptr %s is hostPointer\n", rnk, name);

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_pointer()\n");
#endif
}

void PM::dev_barrier()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_barrier()\n");
#endif

  dpct::get_current_device().queues_wait_and_throw();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_barrier()\n");
#endif
}

int PM::dev_stream_create()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create()\n");
#endif

#if 1
  // just return id of current main stream

  int id = current_queue_id;
#else
  cudaStream_t s;

  cudaStreamCreate(&s);
  
  my_queues.push_back(s);

  int id = my_queues.size() - 1;
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
  return id;
}

void PM::dev_stream_create(dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create(s)\n");
#endif

#if 1
  s = *current_queue;
#else
  cudaStreamCreate(&s);
  _CUDA_CHECK_ERRORS();

  my_queues.push_back(s);
#endif
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
}

void PM::dev_stream_destroy()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy()\n");
#endif

#if 1

#else
  int id = current_queue_id;
  
  cudaStreamDestroy(my_queues[id]);
  _CUDA_CHECK_ERRORS();

  my_queues[id] = NULL;
#endif
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_destroy(dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy(s)\n");
#endif

#if 1

#else
  cudaStreamDestroy(s);
  _CUDA_CHECK_ERRORS();
#endif
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_wait()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif

  *current_queue->wait();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

void PM::dev_stream_wait(dpct::queue_ptr &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif

  s->wait();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

void PM::dev_set_queue(int id)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_set_queue()\n");
#endif

  current_queue = &(my_queues[id]);
  current_queue_id = id;

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_set_queue()\n");
#endif
}

dpct::queue_ptr *PM::dev_get_queue()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_queue()\n");
#endif

  dpct::queue_ptr *q = current_queue;

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_get_queue()\n");
#endif

  return q;
}

/* ---------------------------------------------------------------------- */

void PM::dev_profile_start(const char * label)
{
#ifdef _USE_NVTX
  nvtxRangePushA(label);
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_profile_stop()
{
#ifdef _USE_NVTX
  nvtxRangePop();
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_profile_next(const char * label)
{
#ifdef _USE_NVTX
  nvtxRangePop();
  nvtxRangePushA(label);
#endif
}

#endif
