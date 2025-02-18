#if defined(_USE_CPU)

#ifndef PM_HOST_H
#define PM_HOST_H

#include <iostream>
#include <vector>

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
    void* dev_malloc_async(size_t, void *);
    void* dev_malloc_host(size_t);

    void dev_free(void*);
    void dev_free_async(void*, void *);
    void dev_free_host(void*);

    void dev_push(void*, void*, size_t);
    void dev_pull(void*, void*, size_t);
    void dev_copy(void*, void*, size_t);

    void dev_barrier();

    int dev_push_async(void *, void *, size_t, void *);
    void dev_pull_async(void *, void *, size_t, void *);
			
    void dev_check_pointer(int, const char *, void *);

    int dev_stream_create();
    void dev_stream_create(void *);
    void dev_stream_destroy();
    void dev_stream_destroy(void *);
    void dev_stream_wait(void *);

    void dev_set_queue(int);
    void * dev_get_queue();
    
  private:
    
    void uuid_print(size_t);

    std::vector<void *> my_queues;
    void * current_queue;
    int current_queue_id;
  };

}

#endif

#endif
