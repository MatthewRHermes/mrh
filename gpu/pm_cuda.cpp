#if defined(_GPU_CUDA)

#include <stdio.h>
#include <iostream>

#include <iomanip>
#include <vector>
#include <tuple>

#include "pm_cuda.h"

//https://stackoverflow.com/questions/68823023/set-cuda-device-by-uuid
void uuid_print(cudaUUID_t a){
  std::cout << "GPU";
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  for (auto t : r){
    std::cout << "-";
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)a.bytes[i];
  }
  std::cout << std::endl;
}
/*
#define _CUDA_CHECK_ERRORS()               \
{                                          \
  cudaError err = cudaGetLastError();      \
  if(err != cudaSuccess) {                 \
    std::cout                              \
      << "CUDA error with code "           \
      << cudaGetErrorString(err)           \
      << " in file " << __FILE__           \
      << " at line " << __LINE__           \
      << ". Exiting...\n";                 \
    exit(1);                               \
  }                                        \
}
*/

int dev_num_devices()
{
  int num_devices;

  cudaGetDeviceCount(&num_devices);
  _CUDA_CHECK_ERRORS();
  
  return num_devices;
}

void dev_properties(int ndev)
{
  for(int i=0; i<ndev; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    _CUDA_CHECK_ERRORS();

    char name[256];
    strcpy(name, prop.name);

    printf("  [%i] Platform[ Nvidia ] Type[ GPU ] Device[ %s ]  uuid= ", i, name);
    uuid_print(prop.uuid);
    printf("\n");
  }

}

int dev_check_peer(int rank, int ngpus)
{
  int err = 0;
  if(rank == 0) printf("\nChecking P2P Access\n");
  for(int ig=0; ig<ngpus; ++ig) {
    cudaSetDevice(ig);
    //if(rank == 0) printf("Device i= %i\n",ig);

    int n = 1;
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) {
        int access;
        cudaDeviceCanAccessPeer(&access, ig, jg);
        n += access;

        //if(rank == 0) printf("  --  Device j= %i  access= %i\n",jg,access);
      }
    }
    if(n != ngpus) err += 1;
  }

  return err;
}

void dev_set_device(int id)
{
  cudaSetDevice(id);
  _CUDA_CHECK_ERRORS();
}

int dev_get_device()
{
  int id;
  cudaGetDevice(&id);
  _CUDA_CHECK_ERRORS();
  return id;
}

void * dev_malloc(size_t N)
{
  void * ptr;
  cudaMalloc((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  return ptr;
}

void * dev_malloc_host(size_t N)
{
  void * ptr;
  cudaMallocHost((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  return ptr;
}

void dev_free(void * ptr)
{
  cudaFree(ptr);
  _CUDA_CHECK_ERRORS();
}

void dev_free_host(void * ptr)
{
  cudaFreeHost(ptr);
  _CUDA_CHECK_ERRORS();
}

void dev_push(void * d_ptr, void * h_ptr, size_t N)
{
  cudaMemcpy(d_ptr, h_ptr, N, cudaMemcpyHostToDevice);
  _CUDA_CHECK_ERRORS();
}

void dev_pull(void * d_ptr, void * h_ptr, size_t N)
{
  cudaMemcpy(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost);
  _CUDA_CHECK_ERRORS();
}

void dev_copy(void * dest, void * src, size_t N)
{
  printf("correct usage dest vs. src??]n");
  cudaMemcpy(dest, src, N, cudaMemcpyDeviceToDevice);
  _CUDA_CHECK_ERRORS();
}

void dev_check_pointer(int rnk, const char * name, void * ptr)
{
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);
  if(attributes.devicePointer != NULL) printf("(%i) ptr %s is devicePointer\n",rnk,name);
  if(attributes.hostPointer != NULL) printf("(%i) ptr %s is hostPointer\n",rnk,name);
}

#endif
