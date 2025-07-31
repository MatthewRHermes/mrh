#if defined(_GPU_CUDA)

#include <stdio.h>
#include <iostream>

#include <iomanip>
#include <vector>
#include <tuple>

#include "pm.h"

//#define _DEBUG_PM

using namespace PM_NS;

/* ---------------------------------------------------------------------- */

PM::PM()
{
  int num_devices = dev_num_devices();

  // initialize main queue/stream for each device
  
  for(int i=0; i<num_devices; ++i) {
    cudaSetDevice(i);
    _CUDA_CHECK_ERRORS();

    cudaStream_t s;
    cudaStreamCreate(&s);

    my_queues.push_back(s);
  }

  cudaSetDevice(0);
}

/* ---------------------------------------------------------------------- */

PM::~PM()
{
  int n = my_queues.size();
  for (int i=0; i<n; ++i) cudaStreamDestroy(my_queues[i]);
}

/* ---------------------------------------------------------------------- */

//https://stackoverflow.com/questions/68823023/set-cuda-device-by-uuid
void PM::uuid_print(cudaUUID_t a){
  std::cout << "GPU";
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  for (auto t : r){
    std::cout << "-";
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)a.bytes[i];
  }
  std::cout << std::endl;
}

/* ---------------------------------------------------------------------- */

int PM::dev_num_devices()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_num_devices()\n");
#endif
  int num_devices;
  
  cudaGetDeviceCount(&num_devices);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_num_devices() : num_devices= %i\n",num_devices);
#endif
  
  return num_devices;
}

/* ---------------------------------------------------------------------- */

void PM::dev_properties(int ndev)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_properties()\n");
#endif
  
  for(int i=0; i<ndev; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    _CUDA_CHECK_ERRORS();

    char name[256];
    strcpy(name, prop.name);

    printf("LIBGPU ::  [%i] Platform[ Nvidia ] Type[ GPU ] Device[ %s ]  uuid= ", i, name);
    uuid_print(prop.uuid);
    printf("\n");
  }

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_properties()\n");
#endif
}

/* ---------------------------------------------------------------------- */

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
    cudaSetDevice(ig);
#ifdef _DEBUG_PM
    if(rank == 0) printf("LIBGPU: -- Device i= %i\n",ig);
#endif

    int n = 1;
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) {
        int access;
        cudaDeviceCanAccessPeer(&access, ig, jg);
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

/* ---------------------------------------------------------------------- */

void PM::dev_enable_peer(int rank, int ngpus)
{
#ifdef _DEBUG_PM
  if(rank == 0) {
    printf("Inside PM::dev_enable_peer()\n");
    printf("LIBGPU: -- Enabling peer access for ngpus= %i\n",ngpus);
  }
#endif

  for(int ig=0; ig<ngpus; ++ig) {
    cudaSetDevice(ig);
    
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) cudaDeviceEnablePeerAccess(jg, 0);
    }
    
  }
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_enable_peer()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_set_device(int id)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_set_device()\n");
#endif
  
  cudaSetDevice(id);
  _CUDA_CHECK_ERRORS();

  current_queue = &(my_queues[id]);
  current_queue_id = id;

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_set_devices()\n");
#endif
}

/* ---------------------------------------------------------------------- */

int PM::dev_get_device()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_device()\n");
#endif
  
  int id;
  cudaGetDevice(&id);
  _CUDA_CHECK_ERRORS();

  current_queue_id = id; // matches whatever cudaGetDevice() returns

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_get_device() : id= %i\n",id);
#endif
  
  return id;
}

/* ---------------------------------------------------------------------- */

void * PM::dev_malloc(size_t N, std::string name, const char * file, int line)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc()\n");
#endif
  
  profile_memory(N, name, PROFILE_MEM_MALLOC);
  
  void * ptr;
  cudaMalloc((void**) &ptr, N);

  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("LIBGPU :: Error : PM::dev_malloc() failed to allocate %lu bytes for name= %s from file= %s line= %i\n",
	   N,name.c_str(),file,line);
    print_mem_summary();
    exit(1);
  }
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc()\n");
#endif
  
  return ptr;
}

/* ---------------------------------------------------------------------- */

void * PM::dev_malloc_async(size_t N, std::string name, const char * file, int line)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif

  profile_memory(N, name, PROFILE_MEM_MALLOC);
  
  void * ptr;
#ifdef _NO_CUDA_ASYNC
  cudaMalloc((void**) &ptr, N);
#else
  cudaMallocAsync((void**) &ptr, N, *current_queue);
#endif

  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("LIBGPU :: Error : PM::dev_malloc_async() failed to allocate %lu bytes for name= %s from file= %s line= %i\n",
	   N,name.c_str(),file,line);
    print_mem_summary();
    exit(1);
  }
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

/* ---------------------------------------------------------------------- */

void * PM::dev_malloc_async(size_t N, cudaStream_t &s, std::string name, const char * file, int line)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif
  
  profile_memory(N, name, PROFILE_MEM_MALLOC);
  
  void * ptr;
#ifdef _NO_CUDA_ASYNC
  cudaMalloc((void**) &ptr, N);
#else
  cudaMallocAsync((void**) &ptr, N, s);
#endif

  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("LIBGPU :: Error : PM::dev_malloc_async() failed to allocate %lu bytes for name= %s from file= %s line= %i\n",
	   N,name.c_str(),file,line);
    print_mem_summary();
    exit(1);
  }
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

/* ---------------------------------------------------------------------- */

void * PM::dev_malloc_host(size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_host()\n");
#endif
  
  void * ptr;
  cudaMallocHost((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_host()\n");
#endif
  
  return ptr;
}

/* ---------------------------------------------------------------------- */

void PM::dev_free(void * ptr, std::string name)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free()\n");
#endif
  
  profile_memory(0, name, PROFILE_MEM_FREE);
  
  if(ptr) cudaFree(ptr);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_free_async(void * ptr, std::string name)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
 
  profile_memory(0, name, PROFILE_MEM_FREE);
  
#ifdef _NO_CUDA_ASYNC
  if(ptr) cudaFree(ptr);
#else 
  if(ptr) cudaFreeAsync(ptr, *current_queue);
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_free_async(void * ptr, cudaStream_t &s, std::string name)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
 
  profile_memory(0, name, PROFILE_MEM_FREE);
  
#ifdef _NO_CUDA_ASYNC
  if(ptr) cudaFree(ptr);
#else 
  if(ptr) cudaFreeAsync(ptr, s);
#endif
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_free_host(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_host()\n");
#endif
  
  if(ptr) cudaFreeHost(ptr);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_host()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push()\n");
#endif
  
  cudaMemcpy(d_ptr, h_ptr, N, cudaMemcpyHostToDevice);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push()\n");
#endif
}

/* ---------------------------------------------------------------------- */

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif
  
  cudaMemcpyAsync(d_ptr, h_ptr, N, cudaMemcpyHostToDevice, *current_queue);
  _CUDA_CHECK_ERRORS2();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

/* ---------------------------------------------------------------------- */

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif
  
  cudaMemcpyAsync(d_ptr, h_ptr, N, cudaMemcpyHostToDevice, s);
  _CUDA_CHECK_ERRORS2();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

/* ---------------------------------------------------------------------- */

void PM::dev_pull(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull()\n");
#endif
  
  cudaMemcpy(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  cudaMemcpyAsync(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost, *current_queue);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  cudaMemcpyAsync(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost, s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_memcpy_peer(void * d_ptr, int dest, void * s_ptr, int src, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_memcpy_peer()\n");
#endif
  
  cudaMemcpyPeer(d_ptr, dest, s_ptr, src, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_memcpy_peer()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_memcpy_peer_async(void * d_ptr, int dest, void * s_ptr, int src, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_memcpy_peer_async()\n");
#endif
  
  cudaMemcpyPeerAsync(d_ptr, dest, s_ptr, src, N, *current_queue);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_memcpy_peer_async()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_copy(void * dest, void * src, size_t N)
{ 
#ifdef _DEBUG_PM
  printf("Inside PM::dev_copy()\n");
#endif
  
  cudaMemcpy(dest, src, N, cudaMemcpyDeviceToDevice);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_copy()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_check_pointer()\n");
#endif
  
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);
  if(attributes.devicePointer != NULL) printf("(%i) ptr %s is devicePointer\n",rnk,name);
  if(attributes.hostPointer != NULL) printf("(%i) ptr %s is hostPointer\n",rnk,name);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_pointer()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_barrier()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_barrier()\n");
#endif
  
  cudaDeviceSynchronize();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_barrier()\n");
#endif
}

/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */

void PM::dev_stream_create(cudaStream_t & s)
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

/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */

void PM::dev_stream_destroy(cudaStream_t & s)
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

/* ---------------------------------------------------------------------- */

void PM::dev_stream_wait()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif
  
  cudaStreamSynchronize(*current_queue);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void PM::dev_stream_wait(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif
  
  cudaStreamSynchronize(s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */

cudaStream_t * PM::dev_get_queue()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_queue()\n");
#endif

  cudaStream_t * q = current_queue;
  
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

/* ---------------------------------------------------------------------- */

#if defined (_PROFILE_PM_MEM)
void PM::profile_memory(size_t N, std::string name_, int mode)
{
  std::string name = name_ + "-" + std::to_string(current_queue_id);
  //  printf("PM::dev_malloc()  name= %s\n",name.c_str());

  auto it_ = std::find(profile_mem_name.begin(), profile_mem_name.end(), name);

  int indx = it_ - profile_mem_name.begin();

  if(mode == PROFILE_MEM_MALLOC) {
  
    if(indx < profile_mem_name.size()) {
      profile_mem_size[indx] += N;
      profile_mem_count_alloc[indx]++;
      if(N > profile_mem_max_size[indx]) profile_mem_max_size[indx] = N;
    } else {
      profile_mem_name.push_back(name);
      profile_mem_size.push_back(N);
      profile_mem_max_size.push_back(N);
      profile_mem_count_alloc.push_back(1);
      profile_mem_count_free.push_back(0);
    }

  } else if(mode == PROFILE_MEM_FREE) {

    if(indx < profile_mem_name.size()) {
      profile_mem_size[indx] = 0;
      profile_mem_count_free[indx]++;
    }
    
  } else {
    printf("LIBGPU :: Error : Unsupported profile_memory mode= %i  name= %s\n",mode,name.c_str());
    exit(1);
  }
    
}
#else
void PM::profile_memory(size_t N, std::string name_, int mode) {}
#endif

/* ---------------------------------------------------------------------- */

#if defined(_PROFILE_PM_MEM)
void PM::print_mem_summary()
{
  printf("\nLIBGPU :: PROFILE_PM_MEM\n");
  
  double sum_mb = 0.0;
  
  for(int i=0; i<profile_mem_name.size(); ++i) {
    double max_size_mb = profile_mem_max_size[i] / 1024.0 / 1024.0;
    double size_mb = profile_mem_size[i] / 1024.0 / 1024.0;
    
    sum_mb += max_size_mb;
    
    // printf("LIBGPU :: PROFILE_PM_MEM :: [%3i] name= %20s  max_size= %6.1f MBs  current_size= %6.1f MBs  num_alloc= %lu  num_free= %lu\n",
    // 	   i, profile_mem_name[i].c_str(), max_size_mb, size_mb, profile_mem_count_alloc[i], profile_mem_count_free[i]);
    printf("LIBGPU :: PROFILE_PM_MEM :: [%3i] name= %20s  max_size= %6.1f MBs  current_size= %lu bytes  num_alloc= %lu  num_free= %lu\n",
	   i, profile_mem_name[i].c_str(), max_size_mb, profile_mem_size[i], profile_mem_count_alloc[i], profile_mem_count_free[i]);
  }
  
  int num_devices = dev_num_devices();
  printf("LIBGPU :: PROFILE_PM_MEM :: [total]  %6.1f MBs  %6.1f MBs / device\n", sum_mb, sum_mb/(double) num_devices);
}
#else
void PM::print_mem_summary() {};
#endif

/* ---------------------------------------------------------------------- */

#endif
