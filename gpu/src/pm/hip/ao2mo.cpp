/* -*- c++ -*- */

#if defined(_GPU_HIP)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _TRANSPOSE_BLOCK_SIZE 16
#define _TRANSPOSE_NUM_ROWS 16
#define _UNPACK_BLOCK_SIZE 32
#define _HESSOP_BLOCK_SIZE 32
#define _DEFAULT_BLOCK_SIZE 32

#define _ATOMICADD
#define _ACCELERATE_KERNEL

#define _TILE(A,B) (A + B - 1) / B
#define _HIP_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

__global__ void _get_bufd( const double* bufpp, double* bufd, int naux, int nmo){
    
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < naux && j < nmo) {
        bufd[i * nmo + j] = bufpp[(i*nmo + j)*nmo + j];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao, int nmo) {
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncas && j < nao) {
        small_mat[i * nao + j] = big_mat[j*nmo + i+ncore];
    }
}


/* ---------------------------------------------------------------------- */

__global__ void _get_bufpa (const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= naux) return;
  if(j >= nmo) return;
  if(k >= ncas) return;
  
  int inputIndex = (i*nmo + j)*nmo + k+ncore;
  int outputIndex = (i*nmo + j)*ncas + k;
  bufpa[outputIndex] = bufpp[inputIndex];
}

/* ---------------------------------------------------------------------- */

__global__ void _get_bufaa (const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= naux) return;
  if(j >= ncas) return;
  if(k >= ncas) return;
  
  int inputIndex = (i*nmo + (j+ncore))*nmo + k+ncore;
  int outputIndex = (i*ncas + j)*ncas + k;
  bufaa[outputIndex] = bufpp[inputIndex];
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufpa(const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE,_UNPACK_BLOCK_SIZE,1);
  dim3 grid_size (_TILE(naux, block_size.x), _TILE(nmo, block_size.y), ncas);
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _get_bufpa<<<grid_size, block_size, 0, s>>>(bufpp, bufpa, naux, nmo, ncore, ncas);
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufaa(const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE,1,1);
  dim3 grid_size (_TILE(naux, block_size.x), ncas, ncas);
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _get_bufaa<<<grid_size, block_size, 0, s>>>(bufpp, bufaa, naux, nmo, ncore, ncas);
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao, int nmo)
{
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ncas, block_size.x), _TILE(nao, block_size.y));
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _get_mo_cas<<<grid_size, block_size, 0, s>>>(big_mat, small_mat, ncas, ncore, nao, nmo);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::_get_mo_cas :: ncas= %i  nao= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, nao, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufd( const double* bufpp, double* bufd, int naux, int nmo)
{
  dim3 block_size (_UNPACK_BLOCK_SIZE,_UNPACK_BLOCK_SIZE,1);
  dim3 grid_size (_TILE(naux, block_size.x),_TILE(nmo, block_size.y),1);
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _get_bufd<<<grid_size, block_size, 0, s>>>(bufpp, bufd, naux, nmo);
}

/* ---------------------------------------------------------------------- */

#endif