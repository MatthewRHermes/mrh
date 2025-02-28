#if defined(_GPU_HIP)

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
    hipSetDevice(i);
    _HIP_CHECK_ERRORS();

    hipStream_t s;
    hipStreamCreate(&s);

    my_queues.push_back(s);
  }

  hipSetDevice(0);
}

PM::~PM()
{
  int n = my_queues.size();
  for (int i=0; i<n; ++i) hipStreamDestroy(my_queues[i]);
  
}

//https://stackoverflow.com/questions/68823023/set-hip-device-by-uuid
void PM::uuid_print(hipUUID_t a){
  std::cout << "GPU";
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  for (auto t : r){
    std::cout << "-";
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)a.bytes[i];
  }
  std::cout << std::endl;
}

int PM::dev_num_devices()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_num_devices()\n");
#endif
  int num_devices;
  
  hipGetDeviceCount(&num_devices);
  _HIP_CHECK_ERRORS();
  
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
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, i);
    _HIP_CHECK_ERRORS();

    char name[256];
    strcpy(name, prop.name);

    printf("LIBGPU ::  [%i] Platform[ AMD ] Type[ GPU ] Device[ %s ]  uuid= ", i, name);
    uuid_print(prop.uuid);
    //printf("\n");
  }

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_properties()\n");
#endif
}

int PM::dev_check_peer(int rank, int ngpus)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_check_peer()\n");
#endif
  
  int err = 0;
  if(rank == 0) printf("\nChecking P2P Access\n");
  for(int ig=0; ig<ngpus; ++ig) {
    hipSetDevice(ig);
    //if(rank == 0) printf("Device i= %i\n",ig);

    int n = 1;
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) {
        int access;
        hipDeviceCanAccessPeer(&access, ig, jg);
        n += access;

        //if(rank == 0) printf("  --  Device j= %i  access= %i\n",jg,access);
      }
    }
    if(n != ngpus) err += 1;
  }

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_peer()\n");
#endif
  
  return err;
}

void PM::dev_set_device(int id)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_set_device()\n");
#endif
  
  hipSetDevice(id);
  _HIP_CHECK_ERRORS();

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
  hipGetDevice(&id);
  _HIP_CHECK_ERRORS();

  current_queue_id = id; // matches whatever hipGetDevice() returns

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
  hipMalloc((void**) &ptr, N);
  _HIP_CHECK_ERRORS();
  
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
  hipMallocAsync((void**) &ptr, N, *current_queue);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

void * PM::dev_malloc_async(size_t N, hipStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif
  
  void * ptr;
  hipMallocAsync((void**) &ptr, N, s);
  _HIP_CHECK_ERRORS();
  
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
  hipHostMalloc((void**) &ptr, N, hipHostMallocDefault);
  _HIP_CHECK_ERRORS();
  
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
  
  if(ptr) hipFree(ptr);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free()\n");
#endif
}

void PM::dev_free_async(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
  
  if(ptr) hipFreeAsync(ptr, *current_queue);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_async(void * ptr, hipStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
  
  if(ptr) hipFreeAsync(ptr, s);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_host(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_host()\n");
#endif
  
  if(ptr) hipHostFree(ptr);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_host()\n");
#endif
}

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push()\n");
#endif
  
  hipMemcpy(d_ptr, h_ptr, N, hipMemcpyHostToDevice);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push()\n");
#endif
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif
  
  hipMemcpyAsync(d_ptr, h_ptr, N, hipMemcpyHostToDevice, *current_queue);
  _HIP_CHECK_ERRORS2();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N, hipStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif
  
  hipMemcpyAsync(d_ptr, h_ptr, N, hipMemcpyHostToDevice, s);
  _HIP_CHECK_ERRORS2();
  
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
  
  hipMemcpy(h_ptr, d_ptr, N, hipMemcpyDeviceToHost);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull()\n");
#endif
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  hipMemcpyAsync(h_ptr, d_ptr, N, hipMemcpyDeviceToHost, *current_queue);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N, hipStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  hipMemcpyAsync(h_ptr, d_ptr, N, hipMemcpyDeviceToHost, s);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

void PM::dev_copy(void * dest, void * src, size_t N)
{ 
#ifdef _DEBUG_PM
  printf("Inside PM::dev_copy()\n");
#endif
  
  printf("correct usage dest vs. src??]n");
  hipMemcpy(dest, src, N, hipMemcpyDeviceToDevice);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_copy()\n");
#endif
}

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_check_pointer()\n");
#endif
  
  hipPointerAttribute_t attributes;
  hipPointerGetAttributes(&attributes, ptr);
  if(attributes.devicePointer != NULL) printf("(%i) ptr %s is devicePointer\n",rnk,name);
  if(attributes.hostPointer != NULL) printf("(%i) ptr %s is hostPointer\n",rnk,name);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_pointer()\n");
#endif
}

void PM::dev_barrier()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_barrier()\n");
#endif
  
  hipDeviceSynchronize();
  _HIP_CHECK_ERRORS();
  
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
  hipStream_t s;

  hipStreamCreate(&s);
  
  my_queues.push_back(s);

  int id = my_queues.size() - 1;
#endif
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
  return id;
}

void PM::dev_stream_create(hipStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create(s)\n");
#endif

#if 1
  s = *current_queue;
#else
  hipStreamCreate(&s);
  _HIP_CHECK_ERRORS();

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
  
  hipStreamDestroy(my_queues[id]);
  _HIP_CHECK_ERRORS();

  my_queues[id] = NULL;
#endif
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_destroy(hipStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy(s)\n");
#endif

#if 1

#else
  hipStreamDestroy(s);
  _HIP_CHECK_ERRORS();
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
  
  hipStreamSynchronize(*current_queue);
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

void PM::dev_stream_wait(hipStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif
  
  hipStreamSynchronize(s);
  _HIP_CHECK_ERRORS();
  
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

hipStream_t * PM::dev_get_queue()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_queue()\n");
#endif

  hipStream_t * q = current_queue;
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_get_queue()\n");
#endif

  return q;
}

#endif
