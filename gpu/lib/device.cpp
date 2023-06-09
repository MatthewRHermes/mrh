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
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  printf("LIBGPU: destroying device\n");
  
  dev_free(d_data);
  
  dev_free(d_partial);
  free(partial);
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
