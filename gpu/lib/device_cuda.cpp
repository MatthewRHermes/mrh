/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "device.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

__global__ void _compute(double * d, int N, double * p)
{
  __shared__ double cache[_SIZE_BLOCK];
  
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int cache_id = threadIdx.x;

  // useful work
  
  double temp = 0.0;
  while (id < N) {
    temp += d[id];
    d[id] += 1.0;
    
    id += blockDim.x * gridDim.x;
  }

  // set thread-local value
  
  cache[cache_id] = temp;

  // block
  
  __syncthreads();

  // manually reduce values from threads within block to master thread's value
  
  int i=blockDim.x / 2;
  while(i != 0) {
    if(cache_id < i) cache[cache_id] += cache[cache_id + i];
    __syncthreads();
    i /= 2;
  }

  // store master thread's value in global array for host
  
  if(cache_id == 0) p[blockIdx.x] = cache[0];
}

/* ---------------------------------------------------------------------- */

double Device::compute(double * data)
{ 
  // update data on gpu

  dev_push(d_data, data, size_data);
  
  // do something useful
    
  _compute<<<grid_size, block_size>>>(d_data,n,d_partial);
  _CUDA_CHECK_ERRORS();

  dev_pull(d_data, data, size_data);
  dev_pull(d_partial, partial, grid_size*sizeof(double));
    
  double sum = 0.0;
  for(int i=0; i<grid_size; ++i) sum+= partial[i];
    
  printf(" C-Kernel : N= %i  sum= %f\n",n, sum);

  return sum;
}

/* ---------------------------------------------------------------------- */

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> ppaa, py::array_t<double> papa, py::array_t<double> paaa,
			      py::array_t<double> ocm2, py::array_t<double> tcm2, py::array_t<double> gorb,
			      int ncore, int nocc, int nmo)
{
  printf("Device::orbital_response w/ CUDA backend not yet enabled\n");
}

#endif
