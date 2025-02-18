#if defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)

#include <stdio.h>
#include <iostream>

#include <iomanip>
#include <vector>
#include <tuple>

#include "pm.h"

//#define _DEBUG_PM
#define _ENABLE_INTEL_CCS

using namespace PM_NS;

PM::PM()
{
  current_queue = nullptr;
  current_queue_id = -1;

  // initialize main queue/stream for each device as in-order

  sycl::property_list q_prop{sycl::property::queue::in_order()};

#if defined(_ENABLE_INTEL_CCS) // expecting Intel PVC GPUs on Aurora
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  int num_plat = 0;
  int num_dev = 0;
  
  for (const auto &plat : platforms) {   
    std::vector<sycl::device> devices = plat.get_devices();
    
    for (const auto &dev : devices) {

      auto part_prop = dev.get_info<sycl::info::device::partition_properties>();

      // part_prop will show numa domains for single GPU
      // part_prop will be empty for single tile if ZEX_NUMBER_OF_CCS not set
      // part_prop will show cslice for single tile if ZEX_NUMBER_OF_CCS set
	
      if(part_prop.empty()) {
	
	sycl::queue q(dev, q_prop);
	my_queues.push_back(q);	
	num_dev++;
	
      } else {
	
	for(int i=0; i<part_prop.size(); ++i) {
	  // can device be partitioned into cslices?
	
	  if(part_prop[i] == sycl::info::partition_property::partition_by_affinity_domain) {
	    
	    sycl::queue q(dev, q_prop);
	    my_queues.push_back(q);
	    num_dev++;
	    
	    break;
	  } else if(part_prop[i] == sycl::info::partition_property::ext_intel_partition_by_cslice) {
	    auto sub_devices = dev.create_sub_devices<sycl::info::partition_property::ext_intel_partition_by_cslice>();
	  
	    for(int j=0; j<sub_devices.size(); ++j) {
	    
	      sycl::queue q(sub_devices[j], q_prop);
	      my_queues.push_back(q);	      
	      num_dev++;
	      
	    }
	    break;
	  } // if(partition_by_affinity_domain || cslice)
	} // for(part_prop.size())
      } // if(prop_prop.empty)

    } // for(dev)
    num_plat++;
  } // for(plat)

  num_devices = num_dev;
#else
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  num_devices = 0;
  
  for (const auto &plat : platforms) {
    std::vector<sycl::device> devices = plat.get_devices();

    num_devices += devices.size();
  
    for (const auto &dev : devices) {
      sycl::queue q(dev, q_prop);
      if(dev.is_gpu()) my_queues.push_back(q);
    }
    
  }
#endif
}

void PM::uuid_print(std::array<unsigned char, 16>  a){
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  int first = 1;
  for (auto t : r){
    if(!first) std::cout << "-";
    first = 0;
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)a[i];
  }
}

int PM::dev_num_devices()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_num_devices()\n");
#endif
  
  // std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  // num_devices = 0;
  
  // for (const auto &plat : platforms) {
  //   std::vector<sycl::device> devices = plat.get_devices();

  //   for (const auto &dev : devices) {
  //     sycl::queue q(dev);
  //     if(dev.is_gpu()) num_devices++;
  //   }
  // }
  
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
  
  std::cout << "LIBGPU :: List Platforms and Devices" << std::endl;
  
#if defined(_ENABLE_INTEL_CCS)  
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  int num_plat = 0;
  
  for (const auto &plat : platforms) {
    std::cout << "LIBGPU :: [" << num_plat << "] Platform Name[ "
	      << plat.get_info<sycl::info::platform::name>() << " ] Vendor ["
	      << plat.get_info<sycl::info::platform::vendor>() << " ] Version [ "
	      << plat.get_info<sycl::info::platform::version>() << " ]" << std::endl;
    
    std::vector<sycl::device> devices = plat.get_devices();

    //    std::cout << "# of devices= " << devices.size() << std::endl;
    
    int num_dev = 0;
    for (const auto &dev : devices) {

      auto part_prop = dev.get_info<sycl::info::device::partition_properties>();

      // part_prop will show numa domains for single GPU
      // part_prop will be empty for single tile if ZEX_NUMBER_OF_CCS not set
      // part_prop will show cslice for single tile if ZEX_NUMBER_OF_CCS set
      
      //std::cout << " # of partition propertiess= " << part_prop.size() << std::endl;
      
      if(part_prop.empty()) {
	//std::cout << "No partition properties\n";
	
	sycl::queue q(dev);	  
	std::cout << "LIBGPU :: -- [" << num_dev << "] Device[ "
		  << dev.get_info<sycl::info::device::name>() << " ] Type[ "
		  << (dev.is_gpu() ? "GPU" : "CPU") << " Tile ] Device[ ";
	
	if(dev.is_gpu()) uuid_print(q.get_device().get_info<sycl::ext::intel::info::device::uuid>());
	std::cout << " ]" << std::endl;
	
	num_dev++;
	
      } else {
	
	for(int i=0; i<part_prop.size(); ++i) {
	  // can device be partitioned into cslices
	
	  if(part_prop[i] == sycl::info::partition_property::partition_by_affinity_domain) {
	    auto sub_devices = dev.create_sub_devices<
	      sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
	    
	    //std::cout << "# of domain sub_devices= " << sub_devices.size() << std::endl;

	    sycl::queue q(dev);	  
	    std::cout << "LIBGPU :: -- [" << num_dev << "] Device[ "
		      << dev.get_info<sycl::info::device::name>() << " ] Type[ "
		      << (dev.is_gpu() ? "GPU" : "CPU") << " " << sub_devices.size() << "-Tile ] Device[ ";
	    
	    if(dev.is_gpu()) uuid_print(q.get_device().get_info<sycl::ext::intel::info::device::uuid>());
	    std::cout << " ]" << std::endl;
	    
	    num_dev++;
	    
	    break;
	  } else if(part_prop[i] == sycl::info::partition_property::ext_intel_partition_by_cslice) {
	    auto sub_devices = dev.create_sub_devices<sycl::info::partition_property::ext_intel_partition_by_cslice>();
	  
	    //std::cout << "# of cslice sub_devices= " << sub_devices.size() << std::endl;
	    for(int j=0; j<sub_devices.size(); ++j) {
	    
	      sycl::queue q(sub_devices[j]);
	      std::cout << "LIBGPU :: -- [" << num_dev << "] Device[ "
			<< sub_devices[j].get_info<sycl::info::device::name>() << " ] Type[ "
			<< (sub_devices[j].is_gpu() ? "GPU" : "CPU") << " CCS ] Device[ ";
	    
	      if(sub_devices[j].is_gpu()) uuid_print(q.get_device().get_info<sycl::ext::intel::info::device::uuid>());
	      std::cout << " ]" << std::endl;
	      
	      num_dev++;
	    }
	    break;
	  } else {
	    //std::cout << "No ext_intel_partition_by_cslice or partition_by_affinity_domain\n";
	    
	  } // if(partition_by_affinity_domain || cslice)
	} // for(part_prop.size())
      }

    }
    num_plat++;
  }
#else
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  int num_plat = 0;
  
  for (const auto &plat : platforms) {
    std::cout << "LIBGPU :: [" << num_plat << "] Platform Name[ "
	      << plat.get_info<sycl::info::platform::name>() << " ] Vendor ["
	      << plat.get_info<sycl::info::platform::vendor>() << " ] Version [ "
	      << plat.get_info<sycl::info::platform::version>() << " ]" << std::endl;

    std::vector<sycl::device> devices = plat.get_devices();

    int num_dev = 0;
    for (const auto &dev : devices) {
      sycl::queue q(dev);
      std::cout << "LIBGPU :: -- [" << num_dev << "] Device[ "
		<< dev.get_info<sycl::info::device::name>() << " ] Type[ "
		<< (dev.is_gpu() ? "GPU" : "CPU") << " ] Device[ ";
      
      if(dev.is_gpu()) uuid_print(q.get_device().get_info<sycl::ext::intel::info::device::uuid>());
      std::cout << " ]" << std::endl;
      num_dev++;
    }
    num_plat++;
  }
#endif
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_properties()\n");
#endif
}

int PM::dev_check_peer(int rank, int ngpus)
{
// #ifdef _DEBUG_PM
//   printf("Inside PM::dev_check_peer()\n");
// #endif
  
//   int err = 0;
//   if(rank == 0) printf("\nChecking P2P Access\n");
//   for(int ig=0; ig<ngpus; ++ig) {
//     cudaSetDevice(ig);
//     //if(rank == 0) printf("Device i= %i\n",ig);

//     int n = 1;
//     for(int jg=0; jg<ngpus; ++jg) {
//       if(jg != ig) {
//         int access;
//         cudaDeviceCanAccessPeer(&access, ig, jg);
//         n += access;

//         //if(rank == 0) printf("  --  Device j= %i  access= %i\n",jg,access);
//       }
//     }
//     if(n != ngpus) err += 1;
//   }

// #ifdef _DEBUG_PM
//   printf(" -- Leaving PM::dev_check_peer()\n");
// #endif
  
  return 1;
}

void PM::dev_check_errors()
{
  _SYCL_CHECK_ERRORS()
}

void PM::dev_set_device(int id)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_set_device()\n");
#endif

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

  int id = current_queue_id;

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
  
  void * ptr = (void *) sycl::malloc_device<char>(N, *current_queue);
  current_queue->wait();
  
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
  
  void * ptr = sycl::malloc_device<char>(N, *current_queue);
  //  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

void * PM::dev_malloc_async(size_t N, sycl::queue &q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif
  
  void * ptr = sycl::malloc_device<char>(N, q);
  
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

  if(current_queue) {
    ptr = sycl::malloc_host<char>(N, *current_queue);
    current_queue->wait(); // needed?
  } else {
    ptr = malloc(N);
  }
  
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

  sycl::free(ptr, *current_queue);
  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free()\n");
#endif
}

void PM::dev_free_async(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
  
  sycl::free(ptr, *current_queue);
  //  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_async(void * ptr, sycl::queue &q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
  
  sycl::free(ptr, q);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_host(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_host()\n");
#endif
  
  sycl::free(ptr, *current_queue);
  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_host()\n");
#endif
}

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push()\n");
#endif

  current_queue->memcpy(d_ptr, h_ptr, N);
  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push()\n");
#endif
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif

  current_queue->memcpy(d_ptr, h_ptr, N);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N, sycl::queue &q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif

  q.memcpy(d_ptr, h_ptr, N);
  
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
  
  current_queue->memcpy(h_ptr, d_ptr, N);
  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull()\n");
#endif
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  current_queue->memcpy(h_ptr, d_ptr, N);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N, sycl::queue &q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  q.memcpy(h_ptr, d_ptr, N);
  
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
  current_queue->memcpy(dest, src, N);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_copy()\n");
#endif
}

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
// #ifdef _DEBUG_PM
//   printf("Inside PM::dev_check_pointer()\n");
// #endif
  
//   cudaPointerAttributes attributes;
//   cudaPointerGetAttributes(&attributes, ptr);
//   if(attributes.devicePointer != NULL) printf("(%i) ptr %s is devicePointer\n",rnk,name);
//   if(attributes.hostPointer != NULL) printf("(%i) ptr %s is hostPointer\n",rnk,name);
  
// #ifdef _DEBUG_PM
//   printf(" -- Leaving PM::dev_check_pointer()\n");
// #endif
}

void PM::dev_barrier()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_barrier()\n");
#endif

  current_queue->wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_barrier()\n");
#endif
}

int PM::dev_stream_create()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create()\n");
#endif

  // return stream that corresponds to sycl current queue
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
  return current_queue_id;
}

void PM::dev_stream_destroy()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy()\n");
#endif
  
  // don't think we should destroy any queues at this point
  
  // cudaStreamDestroy(s);
  // _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_wait()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif

  current_queue->wait();
  // cudaStreamSynchronize(s);
  // _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

#if defined(_GPU_SYCL_CUDA)

void PM::dev_stream_create(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create()\n");
#endif

  // return stream that corresponds to current sycl queue
  s = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*current_queue);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
}

void PM::dev_stream_destroy(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy()\n");
#endif
  
  // don't think we should destroy any queues at this point
  
  // cudaStreamDestroy(s);
  // _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_wait(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif

  current_queue->wait();
  // cudaStreamSynchronize(s);
  // _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

#else

void PM::dev_stream_create(sycl::queue & q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create()\n");
#endif

  // just return queue already created
  q = *current_queue;
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
}

void PM::dev_stream_destroy(sycl::queue & q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy()\n");
#endif

  // don't think we should destroy any sycl queues at this point
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_wait(sycl::queue & q)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif

  q.wait();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}
#endif

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

sycl::queue * PM::dev_get_queue()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_queue()\n");
#endif

  sycl::queue * q = current_queue;
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_get_queue()\n");
#endif

  return q;
}

#endif
