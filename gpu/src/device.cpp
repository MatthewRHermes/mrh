/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"


/* ---------------------------------------------------------------------- */

Device::Device()
{
  printf("LIBGPU: created device\n");
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
  
#ifdef _SIMPLE_TIMER
  t_array = (double *) malloc(14 * sizeof(double));
  for(int i=0; i<14; ++i) t_array[i] = 0.0;
  
  t_array_jk = (double* ) malloc(9 * sizeof(double));
  for(int i=0; i<9; ++i) t_array_jk[i] = 0.0;
#endif
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  printf("LIBGPU: destroying device\n");
  
  dev_free(d_data);
  
  dev_free(d_partial);
  free(partial);

#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  free(rho);
  free(vj);
  free(_vktmp);

  free(buf_tmp);
  free(buf3);
  free(buf4);

  free(buf_fdrv);

#ifdef _SIMPLE_TIMER
  t_array_jk[8] += omp_get_wtime() - t0;
#endif
  
#ifdef _SIMPLE_TIMER
  printf("LIBGPU::orbital_response\n");
  double total = 0.0;
  for(int i=0; i<14; ++i) {total += t_array[i]; printf("i= %i  t_array= %f s\n",i,t_array[i]); }
  printf("  total= %f\n",total);

  printf("LIBGPU::get_jk\n");
  total = 0.0;
  for(int i=0; i<9; ++i) {total += t_array_jk[i]; printf("i= %i  t_array= %f s\n",i,t_array_jk[i]); }
  printf("  total= %f\n",total);
  
  free(t_array);
  free(t_array_jk);
#endif
}

/* ---------------------------------------------------------------------- */

int Device::get_num_devices()
{
  printf("LIBGPU: getting number of devices\n");
  return dev_num_devices();
}

/* ---------------------------------------------------------------------- */
    
void Device::get_dev_properties(int N)
{
  printf("LIBGPU: reporting device properties N= %i\n",N);
  dev_properties(N);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_device(int id)
{
  printf("LIBGPU: setting device id= %i\n",id);
  dev_set_device(id);
}

/* ---------------------------------------------------------------------- */

double Device::host_compute(double * data)
{
  double sum = 0.0;
  for(int i=0; i<n; ++i) {
    sum += data[i];
    data[i] += 1.0;
  }
  return sum;
}

/* ---------------------------------------------------------------------- */

void Device::setup(double * data, int _n)
{
  printf(" n= %i\n",_n);

  // save local copies

  n = _n;
  
  // create device buffers

  size_data = n*sizeof(double);
  
  d_data = (double*) dev_malloc(size_data);

  dev_push(d_data, data, size_data);
  
  grid_size = MIN(_SIZE_GRID, (n+_SIZE_BLOCK-1) / _SIZE_BLOCK);
  block_size = _SIZE_BLOCK;

  printf(" Launching kernels w/ grid_size= %lu block_size= %lu\n",grid_size,block_size);

  partial = (double*) malloc(grid_size*sizeof(double));
  d_partial = (double*) dev_malloc(grid_size*sizeof(double));

  double _sum = host_compute(data);
  printf(" C-Kernel Reference E= %f\n",_sum); 
}

/* ---------------------------------------------------------------------- */
