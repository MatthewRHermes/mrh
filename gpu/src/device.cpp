/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

/* ---------------------------------------------------------------------- */

Device::Device()
{
  printf("LIBGPU: created device\n");
  
  pm = new PM();
  
  n = 0;

  d_data = nullptr;

  partial = nullptr;
  d_partial = nullptr;

  size_rho = 0;
  size_vj = 0;
  size_vk = 0;
  size_buf = 0;
  size_fdrv = 0;
  
  rho = nullptr;
  vj = nullptr;
  _vktmp = nullptr;

  buf_tmp = nullptr;
  buf3 = nullptr;
  buf4 = nullptr;

  buf_fdrv = nullptr;
  
#if defined(_GPU_CUDA)
  handle = NULL;
  stream = NULL;
  
  d_buf2 = nullptr;
  d_buf3 = nullptr;
  d_vkk = nullptr;
#endif

  num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();
  
#ifdef _SIMPLE_TIMER
  t_array_count = 0;
  t_array = (double *) malloc(14 * sizeof(double));
  for(int i=0; i<14; ++i) t_array[i] = 0.0;

  t_array_jk_count = 0;
  t_array_jk = (double* ) malloc(9 * sizeof(double));
  for(int i=0; i<9; ++i) t_array_jk[i] = 0.0;
#endif
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  printf("LIBGPU: destroying device\n");
  
  pm->dev_free(d_data);
  
  pm->dev_free(d_partial);
  free(partial);

#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  free(rho);
  free(vj);
  free(_vktmp);

  pm->dev_free_host(buf_tmp);
  pm->dev_free_host(buf3);
  free(buf4);

  free(buf_fdrv);

#ifdef _SIMPLE_TIMER
  t_array_jk[8] += omp_get_wtime() - t0;
#endif
  
#ifdef _SIMPLE_TIMER
  printf("LIBGPU::orbital_response\n");
  double total = 0.0;
  for(int i=0; i<14; ++i) {total += t_array[i]; printf("i= %i  t_array= %f s\n",i,t_array[i]); }
  printf("  total= %f s  count= %i\n",total,t_array_count);

  printf("LIBGPU::get_jk\n");
  total = 0.0;
  for(int i=0; i<9; ++i) {total += t_array_jk[i]; printf("i= %i  t_array= %f s\n",i,t_array_jk[i]); }
  printf("  total= %f s  count= %i\n",total,t_array_jk_count);
  
  free(t_array);
  free(t_array_jk);
#endif

#if defined (_GPU_CUDA)
  nvtxRangePushA("Deallocate");
  pm->dev_free(d_buf2);
  pm->dev_free(d_buf3);
  pm->dev_free(d_vkk);
  nvtxRangePop();  
  
  nvtxRangePushA("Destroy Handle");
  cublasDestroy(handle);
  nvtxRangePop();

  pm->dev_stream_destroy(stream);
#endif

  delete pm;
}

/* ---------------------------------------------------------------------- */

int Device::get_num_devices()
{
  printf("LIBGPU: getting number of devices\n");
  return pm->dev_num_devices();
}

/* ---------------------------------------------------------------------- */
    
void Device::get_dev_properties(int N)
{
  printf("LIBGPU: reporting device properties N= %i\n",N);
  pm->dev_properties(N);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_device(int id)
{
  printf("LIBGPU: setting device id= %i\n",id);
  pm->dev_set_device(id);
}

/* ---------------------------------------------------------------------- */
