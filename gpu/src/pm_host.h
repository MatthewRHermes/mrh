#if defined(_USE_CPU)

#ifndef PM_HOST_H
#define PM_HOST_H

#include <iostream>

namespace PM_NS {

  class PM {

  public:

    PM();
    ~PM() {};

    int num_threads;
    
    int dev_num_devices();
    void dev_properties(int);
    int dev_check_peer(int, int);

    void dev_set_device(int);
    int dev_get_device();

    void* dev_malloc(size_t);
    void* dev_malloc_host(size_t);

    void dev_free(void*);
    void dev_free_host(void*);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    void dev_check_pointer(int, const char *, void *);

  };

}

#endif

#endif
