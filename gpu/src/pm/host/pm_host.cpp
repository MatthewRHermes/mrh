#if defined(_USE_CPU)

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>

#include "../pm.h"

using namespace PM_NS;

PM::PM()
{
}

int PM::dev_num_devices()
{
  return 1;
}

void PM::dev_properties(int device)
{
}

int PM::dev_check_peer(int device, int peer_device)
{
  return 0;
}

void PM::dev_check_errors()
{
}

void PM::dev_set_device(int device)
{
}

int PM::dev_get_device()
{
  return 0;
}

void* PM::dev_malloc(size_t N, std::string name, const char *file, int line)
{
  void * ptr = malloc(N);
  if (!ptr && N) {
    printf("PM::dev_malloc failed: N=%zu (%s %s:%d)\n", N, name.c_str(), file, line);
  }
  return ptr;
}

void* PM::dev_malloc_async(size_t N, std::string name, const char *file, int line)
{
  return dev_malloc(N, name, file, line);
}

void* PM::dev_malloc_host(size_t N)
{
  return malloc(N);
}

void PM::dev_free(void* ptr, std::string name)
{
  free(ptr);
}

void PM::dev_free_async(void* ptr, std::string name)
{
  free(ptr);
}

void PM::dev_free_host(void* ptr)
{
  free(ptr);
}

void PM::dev_memcpy_peer(void* d_ptr, int dev_d, void* h_ptr, int dev_s, size_t N)
{
  memcpy(d_ptr, h_ptr, N);
}

void PM::dev_enable_peer(int device, int peer_device)
{
}

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N)
{
  memcpy(d_ptr, h_ptr, N);
}

void PM::dev_pull(void * d_ptr, void * h_ptr, size_t N)
{
  memcpy(h_ptr, d_ptr, N);
}

void PM::dev_copy(void * d_ptr, void * h_ptr, size_t N)
{
  memcpy(d_ptr, h_ptr, N);
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N)
{
  memcpy(h_ptr, d_ptr, N);
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N)
{
  memcpy(d_ptr, h_ptr, N);
  return 0;
}

void PM::dev_memset_zero(void * d_ptr, size_t N)
{
  memset(d_ptr, 0, N);
}

void PM::dev_barrier()
{
}

void PM::dev_stream_wait()
{
}

void PM::dev_set_queue(int queue_id)
{
}

void * PM::dev_get_queue()
{
  return NULL;
}

void PM::dev_profile_start(const char * name)
{
}

void PM::dev_profile_stop()
{
}

void PM::dev_profile_next(const char * name)
{
}

void PM::dev_profile_erase()
{
}

void PM::dev_check_pointer(int flag, const char * file, void * ptr)
{
}

int PM::dev_stream_create()
{
  return 0;
}

void PM::dev_stream_create(void * q)
{
}

void PM::dev_stream_destroy()
{
}

void PM::dev_stream_destroy(void * q)
{
}

void PM::print_mem_summary()
{
}

#endif
