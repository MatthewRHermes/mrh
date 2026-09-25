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

/* ---------------------------------------------------------------------- */

__global__ void pack_Mwuv(double *in, double *out, int * map,int nao, int ncas,int ncas_pair)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=nao) return;
    if (j>=ncas) return;
    if (k>=ncas*ncas) return;
    int inputIndex = i*ncas*ncas*ncas+j*ncas*ncas+k;
    int outputIndex = j*ncas_pair*nao+i*ncas_pair+map[k];
    out[outputIndex]=in[inputIndex];
}

/* ---------------------------------------------------------------------- */

__global__ void _pack_eri1(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
 //eri1 is out, buf2 is in, we are packing buf2 of shape naux * nao * nao to eri1 of shape naux * nao_pair 
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  const int indx = j * nao;
  //for(int k=0; k<nao; ++k) buf[indx+k] = tril[ map[indx+k] ];  
  for(int k=0; k<nao; ++k) tril[map[indx+k]] = buf[ indx+k ];  
}

/* ---------------------------------------------------------------------- */

void DeviceImpham::pack_eri(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
#if 1
  //dim3 grid_size(naux, _TILE(nao, _UNPACK_BLOCK_SIZE), 1);
  //dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(naux, nao, 1);
  dim3 block_size(1,1, 1);
#else
  dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _pack_eri1<<<grid_size, block_size, 0, s>>>(eri1, buf2, map, naux, nao, nao_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: _pack_eri1 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}


#endif
