#if defined(_USE_CPU)

#ifndef PM_HOST_H
#define PM_HOST_H

#include <iostream>
#include <omp.h>

#define MIN(A,B) ((A) < (B) ? (A) : (B))

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
