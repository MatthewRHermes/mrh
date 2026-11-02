/* -*- c++ -*- */

#if defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)

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

//#define _DEBUG_DEVICE
//#define _DEBUG_H2EFF
//#define _DEBUG_H2EFF2
//#define _DEBUG_H2EFF_DF
//#define _DEBUG_AO2MO

#define _TILE(A,B) (A + B - 1) / B
#define _SYCL_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void pack_Mwuv(double *in, double *out, int * map,int nao, int ncas,int ncas_pair)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);
    if (i>=nao) return;
    if (j>=ncas) return;
    if (k>=ncas*ncas) return;
    int inputIndex = i*ncas*ncas*ncas+j*ncas*ncas+k;
    int outputIndex = j*ncas_pair*nao+i*ncas_pair+map[k];
    out[outputIndex]=in[inputIndex];
}

/* ---------------------------------------------------------------------- */

void _pack_eri1(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
 //eri1 is out, buf2 is in, we are packing buf2 of shape naux * nao * nao to eri1 of shape naux * nao_pair
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

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
  sycl::range<3> grid_size(1, nao, naux);
  sycl::range<3> block_size(1, 1, 1);
#else
  dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif
  sycl::queue * s =  ctx.pm->dev_get_queue();

  {
    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _pack_eri1(eri1, buf2, map, naux, nao, nao_pair);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: _pack_eri1 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
         naux, nao, _UNPACK_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}


#endif
