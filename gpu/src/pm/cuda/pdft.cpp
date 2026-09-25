/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "../../device/device_pdft.h"

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
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

__global__ void _get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= ngrid) return;

    Pi[i] += rho[i] * rho[i];
}

/* ---------------------------------------------------------------------- */

__global__ void _make_gridkern(double * mo_grid, double * gridkern, int ngrid, int ncas)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ngrid) return;
    if(j >= ncas) return;
    if(k >= ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_mo_grid = &(mo_grid[i*ncas]);
    tmp_gridkern[j*ncas+k] = tmp_mo_grid[j]*tmp_mo_grid[k];
}

/* ---------------------------------------------------------------------- */

__global__ void _make_buf_pdft(double * gridkern, double * cascm2, double * out, int ngrid, int ncas)
{
    //TODO: convert this to a dgemm
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ngrid) return;
    if(j >= ncas*ncas) return;
    if(k >= ncas*ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_cascm2 = &(cascm2[j*ncas*ncas]);
    double * tmp_out = &(out[i*ncas*ncas+j]);
    tmp_out[0] += tmp_gridkern[k]*tmp_cascm2[k];
}

/* ---------------------------------------------------------------------- */

__global__ void _make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
    //TODO: convert this to a dgemm
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= ngrid) return;
    if(j >= ncas*ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_buf = &(buf[i*ncas*ncas]);
    double * tmp_Pi = &(Pi[i]);
    tmp_Pi[0] += tmp_gridkern[j]*tmp_buf[j];
}

/* ---------------------------------------------------------------------- */

void DevicePdft::get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(ngrid, block_size.x),1,1);

  cudaStream_t s = *(ctx.pm->dev_get_queue());

  _get_rho_to_Pi<<<grid_size, block_size,0, s>>>(rho, Pi, ngrid);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rho_to_Pi :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ngrid, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_gridkern(double * d_mo_grid, double * d_gridkern, int ngrid, int ncas)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(ngrid, block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));

  cudaStream_t s = *(ctx.pm->dev_get_queue());

  _make_gridkern<<<grid_size, block_size,0,s>>>(d_mo_grid, d_gridkern, ngrid, ncas);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_gridkern :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_buf_pdft(double * gridkern, double * buf, double * cascm2, int ngrid, int ncas)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(ngrid, block_size.x),_TILE(ncas*ncas,block_size.y),_TILE(ncas*ncas,block_size.z));

  cudaStream_t s = *(ctx.pm->dev_get_queue());

  // buf = aij, klij ->akl, gridkern, cascm2
  _make_buf_pdft<<<grid_size, block_size,0,s>>>(gridkern, cascm2, buf, ngrid, ncas);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_gridkern :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  ctx.pm->dev_check_errors();
#endif

}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(ngrid, block_size.x),_TILE(ncas*ncas,block_size.y),1);

  cudaStream_t s = *(ctx.pm->dev_get_queue());

  _make_Pi_final<<<grid_size, block_size,0,s>>>(gridkern, buf, Pi, ngrid, ncas);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_Pi_final; :: Ngrid= %i Ncas =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ngrid, ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  ctx.pm->dev_check_errors();
#endif
}


#endif
