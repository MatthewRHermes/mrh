#if defined(_USE_CPU)

#ifndef PM_HOST_H
#define PM_HOST_H

#include <iostream>
#include <vector>
#include <string>

namespace PM_NS {

  class PM {

  public:

    PM();
    ~PM() {};

    int num_threads;
    
    int dev_num_devices();
    void dev_properties(int);
    int dev_check_peer(int, int);
    void dev_check_errors();

    void dev_set_device(int);
    int dev_get_device();

    void* dev_malloc(size_t, std::string, const char *, int);
    void* dev_malloc_async(size_t, std::string, const char *, int);
    void* dev_malloc_host(size_t);

    void dev_free(void*, std::string = "DEFAULT");
    void dev_free_async(void*, std::string = "DEFAULT");
    void dev_free_host(void*);

    void dev_memcpy_peer(void*, int, void *, int, size_t);
    void dev_enable_peer(int, int);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    void dev_pull_async(void*, void*, size_t);
    int dev_push_async(void *, void *, size_t);
    void dev_memset_zero(void*, size_t);

    void dev_barrier();

    void dev_stream_wait();

    void dev_set_queue(int);
    void * dev_get_queue();

    void dev_profile_start(const char *);
    void dev_profile_stop();
    void dev_profile_next(const char *);
    void dev_profile_erase();

    void dev_check_pointer(int, const char *, void *);

    int dev_stream_create();
    void dev_stream_create(void *);
    void dev_stream_destroy();
    void dev_stream_destroy(void *);

    void print_mem_summary();

  private:
    
    void uuid_print(size_t);

    std::vector<void *> my_queues;
    void * current_queue;
    int current_queue_id;
  };

}

#endif

#endif
