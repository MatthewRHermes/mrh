/* -*- c++ -*- */

#if defined(_GPU_CUDA)

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
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

__global__ void _extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(i >= ncas) return;
  if(j >= ncas) return;
  
  small_mat[i * ncas + j] = big_mat[(i + ncore) * nmo + (j + ncore)];
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_2310(double * in, double * out, int nmo, int ncas) {
    //a.transpose(2,3,1,0)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= nmo) return;
    if(j >= ncas) return;
    if(k >= ncas) return;

    for(int l=0; l<ncas; ++l) {
      int inputIndex = ((i*ncas+j)*ncas+k)*ncas+l;
      int outputIndex = k*ncas*ncas*nmo + l*ncas*nmo + j*nmo + i;
      out[outputIndex] = in[inputIndex];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_210(double * in, double * out, int naux, int nao, int ncas) {
    //Pum->muP
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= naux) return;
    if(j >= ncas) return;
    if(k >= nao) return;

    int inputIndex = i*nao*ncas+j*nao+k;
    int outputIndex = k*ncas*naux  + j*naux + i;
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

__global__ void _pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= nmo) return;
  if(j >= ncas) return;
  if(k >= ncas_pair) return;

  double * out_buf = &(out[(i*ncas + j) * ncas_pair]);
  double * in_buf = &(in[(i*ncas + j) * ncas*ncas]);

  out_buf[ k ] = in_buf[ map[k] ];
}

/* ---------------------------------------------------------------------- */

__global__ void _unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nmo*ncas) return;
  if(j >= ncas*ncas) return;

  double * in_buf = &(in[i * ncas_pair ]);
  double * out_buf = &(out[i * ncas*ncas ]);

  out_buf[j] = in_buf[ map[j] ];
}

/* ---------------------------------------------------------------------- */

__global__ void _pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]=in[j*ncas*nmo + i];

}

/* ---------------------------------------------------------------------- */

__global__ void _pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]+=in[j*ncas*nmo + i]; // this doesn't work because map spans (ncas x ncas) and has duplicate entries
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(_TILE(ncas,block_size.x), _TILE(ncas,block_size.y));
    
  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _extract_submatrix<<<grid_size, block_size, 0, s>>>(big_mat, small_mat, ncas, ncore, nmo);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- extract_submatrix :: ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(nmo*ncas,_UNPACK_BLOCK_SIZE), _TILE(ncas*ncas,_UNPACK_BLOCK_SIZE), 1);

  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _unpack_h2eff_2d<<<grid_size, block_size, 0>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- _unpack_h2eff_2d :: nmo*ncas= %i  ncas*ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo*ncas, ncas*ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::transpose_2310(double * in, double * out, int nmo, int ncas)
{
  dim3 block_size(1,1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(nmo,block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));
  
  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _transpose_2310<<<grid_size, block_size, 0, s>>>(in, out, nmo, ncas);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::transpose_2310 :: nmo= %i  ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo, ncas, _DEFAULT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, 1, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(_TILE(naux,block_size.x), _TILE(ncas,block_size.y), _TILE(nao,block_size.z));
  
  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _transpose_210<<<grid_size,block_size, 0, s>>>(in, out, naux, nao, ncas);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- h2eff_df_contract1::transpose_210 :: naux= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 naux, ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(1, 1, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(nmo, ncas, _TILE(ncas_pair, _DEFAULT_BLOCK_SIZE));
  
  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _pack_h2eff_2d<<<grid_size, block_size, 0, s>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::_pack_h2eff_2d :: nmo= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo, ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(nmo*ncas,block_size.x), _TILE(ncas*ncas,block_size.y));
  
  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _pack_d_vuwM<<<grid_size,block_size, 0, s>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo*ncas, ncas*ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(nmo*ncas,block_size.x), _TILE(ncas*ncas,block_size.y));
  
  cudaStream_t s = *(ctx.pm->dev_get_queue());
  
  _pack_d_vuwM_add<<<grid_size,block_size, 0, s>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM_add :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo*ncas, ncas*ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}


#endif
