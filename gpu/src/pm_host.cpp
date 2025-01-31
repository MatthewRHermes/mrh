#if defined(_USE_CPU)

#include <stdio.h>
#include <iostream>
#include <cstring>

#include <iomanip>
#include <vector>
#include <tuple>

#include "pm.h"

using namespace PM_NS;

PM::PM()
{
}

int PM::dev_num_devices()
{
  num_threads = 0;
  
#pragma omp parallel
  num_threads = omp_get_num_threads();
  
  const int date = _OPENMP;
  
  double version;
  if     (date == 201107) version = 3.1;
  else if(date == 201307) version = 4.0;
  else if(date == 201511) version = 4.5;
  else if(date == 201611) version = 5.0;
  else if(date == 202011) version = 5.1;
  else {
    printf("Error: unknown omp version: omp_data= %i.\n",date);
    exit(1);
  }
  
  printf("\n  Using OPENMP v%3.1f\n", version);
  printf("  num_threads= %i\n",num_threads);
  
  return 1;
}

void PM::dev_properties(int ndev) {}

int PM::dev_check_peer(int rank, int ngpus) {return 0;}

void PM::dev_set_device(int id) {}

int PM::dev_get_device() {return 0;}

void * PM::dev_malloc(size_t N) {return malloc(N);}

void * PM::dev_malloc_async(size_t N, void * q) {return dev_malloc(N);}

void * PM::dev_malloc_host(size_t N) {return malloc(N);}

void PM::dev_free(void * ptr) {free(ptr);}

void PM::dev_free_async(void * ptr, void * q) {dev_free(ptr);}

void PM::dev_free_host(void * ptr) {free(ptr);}

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N) {memcpy(d_ptr, h_ptr, N);}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N, void * q) {dev_push(d_ptr, h_ptr, N); return 0;}

void PM::dev_pull(void * d_ptr, void * h_ptr, size_t N) {memcpy(h_ptr, d_ptr, N);}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N, void * q) {dev_pull(h_ptr, d_ptr, N);}

void PM::dev_copy(void * dest, void * src, size_t N) {memcpy(dest, src, N);}

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
  if(ptr != nullptr) printf("(%i) ptr %s is hostPointer\n",rnk,name);
}

void PM::dev_barrier() {};

int PM::dev_stream_create() {return 0;};

void PM::dev_stream_create(void * q) {};

void PM::dev_stream_destroy() {};

void PM::dev_stream_destroy(void * q) {};

void PM::dev_stream_wait(void * q) {};

void PM::dev_set_queue(int id) {}

void * PM::dev_get_queue() {return nullptr;};

#endif
