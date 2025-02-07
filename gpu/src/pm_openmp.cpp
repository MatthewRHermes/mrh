#if defined(_GPU_OPENMP)

#include <stdio.h>
#include <iostream>

#include "pm.h"

using namespace PM_NS;

PM::PM()
{
}

int PM::dev_num_devices()
{
  int num_devices = omp_get_num_devices();
  bool err = (num_devices < 0);
  _OMP_CHECK_ERRORS(err);
  
  return num_devices;
}

void PM::dev_properties(int ndev)
{
  int num_devices = omp_get_num_devices();
  int default_device = omp_get_default_device();
  int host = omp_get_initial_device();

  int num_teams = -1;
  int num_threads = -1;
  int num_dev_threads = -1;
#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
#pragma omp target teams map(tofrom: num_teams, num_threads)
  {
    num_teams = omp_get_num_teams();
    num_dev_threads = omp_get_num_threads();
  }
  
  const int date = _OPENMP;

  double version;
  if     (date == 201107) version = 3.1;
  else if(date == 201307) version = 4.0;
  else if(date == 201511) version = 4.5;
  else if(date == 201611) version = 5.0;
  else if(date == 201811) version = 5.0;
  else if(date == 202011) version = 5.1;
  else {
    printf("[PM] Error: unknown omp version: omp_data= %i.\n",date);
    exit(1);
  }
  
  printf("\n  Using OPENMP v%3.1f\n",version);
  printf("  num_devices=     %i\n",num_devices);
  printf("  Default device=  %i\n",default_device);
  printf("  Host=            %i\n",host);
  printf("  num_teams=       %i\n",num_teams);
  printf("  num_dev_threads= %i\n",num_dev_threads);
  printf("  num_threads=     %i\n",num_threads);
}

int PM::dev_check_peer(int rank, int ngpus)
{
  int err = 0;
  // if(rank == 0) printf("\nChecking P2P Access\n");
  // for(int ig=0; ig<ngpus; ++ig) {
  //   cudaSetDevice(ig);
  //   //if(rank == 0) printf("Device i= %i\n",ig);

  //   int n = 1;
  //   for(int jg=0; jg<ngpus; ++jg) {
  //     if(jg != ig) {
  //       int access;
  //       cudaDeviceCanAccessPeer(&access, ig, jg);
  //       n += access;

  //       //if(rank == 0) printf("  --  Device j= %i  access= %i\n",jg,access);
  //     }
  //   }
  //   if(n != ngpus) err += 1;
  // }

  return err;
}

void PM::dev_set_device(int id)
{
  omp_set_default_device(id);
//  _OMP_CHECK_ERRORS();
}


int PM::dev_get_device()
{
  return omp_get_device_num();
}

void * PM::dev_malloc(int N)
{
  int gpu = omp_get_default_device();
  void * ptr = omp_target_alloc(N, gpu);
  bool err = (ptr == nullptr);
  if(err) printf("dev_malloc() failed to allocate gpu= %i  N= %i\n",gpu,N);
  _OMP_CHECK_ERRORS(err);
  //printf("PM::dev_malloc() allocated N= %i  ptr= %p  gpu= %i\n",N,ptr,gpu);
  return ptr;
}

void * PM::dev_malloc_host(int N)
{
  void * ptr = omp_alloc(N, omp_default_mem_alloc);
  bool err = (ptr == nullptr);
  if(err) printf("dev_malloc_host() failed to allocate N= %i\n",N);
  _OMP_CHECK_ERRORS(err);
  return ptr;
}

void PM::dev_free(void * ptr)
{
  int id = omp_get_default_device();
  omp_target_free(ptr, id);
  //_OMP_CHECK_ERRORS();
}

void PM::dev_free_host(void * ptr)
{
  omp_free(ptr, omp_default_mem_alloc);
  //_OMP_CHECK_ERRORS();
}

void PM::dev_push(void * d_ptr, void * h_ptr, int N)
{
  int gpu = omp_get_default_device();
  int host = omp_get_initial_device();
  //printf("PM::dev_push() transferring N= %i  d_ptr= %p  h_ptr= %p  gpu= %i  host= %i\n",N,d_ptr,h_ptr,gpu,host);
  int err = omp_target_memcpy(d_ptr, h_ptr, N, 0, 0, gpu, host);
  if(err) printf("dev_push() : err= %i  d_ptr= %p  h_ptr= %p  N= %i\n",err, &d_ptr, &h_ptr, N);
  _OMP_CHECK_ERRORS(err);
}

void PM::dev_pull(void * d_ptr, void * h_ptr, int N)
{
  int gpu = omp_get_default_device();
  int host = omp_get_initial_device();
  //printf("PM::dev_pull() transferring N= %i  d_ptr= %p  h_ptr= %p  gpu= %i  host= %i\n",N,d_ptr,h_ptr,gpu,host);
  int err = omp_target_memcpy(h_ptr, d_ptr, N, 0, 0, host, gpu);
  if(err) printf("dev_pull() : err= %i  d_ptr= %p  h_ptr= %p  N= %i\n",err, &d_ptr, &h_ptr, N);
  _OMP_CHECK_ERRORS(err);
}

void PM::dev_copy(void * a_ptr, void * b_ptr, int N)
{
  int gpu = omp_get_default_device();
  int err = omp_target_memcpy(b_ptr, a_ptr, N, 0, 0, gpu, gpu);
  if(err) printf("dev_copy() : err= %i  a_ptr= %p  b_ptr= %p  N= %i\n",err, &a_ptr, &b_ptr, N);
  _OMP_CHECK_ERRORS(err);
}

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
  //if(ptr != nullptr) printf("(%i) ptr %s is hostPointer\n",rnk,name);
}

void PM::dev_barrier()
{
  cudaDeviceSynchronize();
}

void PM::dev_push_async(void * d_ptr, void * h_ptr, int N, void * s)
{
  int gpu = omp_get_default_device();
  int host = omp_get_initial_device();
  int err = omp_target_memcpy(d_ptr, h_ptr, N, 0, 0, gpu, host);
  //int err = omp_target_memcpy_async(d_ptr, h_ptr, N, 0, 0, gpu, host, 1, s);
  if(err) printf("dev_push_async() : err= %i  gpu= %i  host= %i  d_ptr= %p  h_ptr= %p  N= %i\n",err, gpu, host, &d_ptr, &h_ptr, N);
  _OMP_CHECK_ERRORS(err);
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, int N, void * s)
{
  int gpu = omp_get_default_device();
  int host = omp_get_initial_device();
  int err = omp_target_memcpy(h_ptr, d_ptr, N, 0, 0, host, gpu);
  //int err = omp_target_memcpy_async(h_ptr, d_ptr, N, 0, 0, host, gpu, 1, s);
  if(err) printf("dev_pull_async() : err= %i  gpu= %i  host= %i  d_ptr= %p  h_ptr= %p  N= %i\n",err,gpu, host, &d_ptr, &h_ptr, N);
  _OMP_CHECK_ERRORS(err);
}

void PM::dev_stream_create(cudaStream_t & s)
{
//  cudaStreamCreate(&s);
  int gpu = omp_get_default_device();
  int nowait = 0;
  s = (cudaStream_t) ompx_get_cuda_stream(gpu, nowait);
  _CUDA_CHECK_ERRORS();
}

void PM::dev_stream_destroy(cudaStream_t & s)
{
  cudaStreamDestroy(s);
  _CUDA_CHECK_ERRORS();
}

void PM::dev_stream_wait(cudaStream_t & s)
{
  cudaStreamSynchronize(s);
  _CUDA_CHECK_ERRORS();
}

#endif
