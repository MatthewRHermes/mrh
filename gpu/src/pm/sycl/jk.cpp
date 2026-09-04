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

// DEPRECATED: legacy integral engine (fdrv helper) -- commented out. See refactor_plan.md.
#if 0
void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf) // this needs to be removed when host+sycl backends ready
{
//   const int ij_pair = nao * nao;
//   const int nao2 = nao * (nao + 1) / 2;
    
// #pragma omp parallel for
//   for (int i = 0; i < nij; i++) {
//     double * buf = &(_buf[i * nao * nao]);

//     int _i, _j, _ij;
//     double * tril = vin + nao2*i;
//     for (_ij = 0, _i = 0; _i < nao; _i++) 
//       for (_j = 0; _j <= _i; _j++, _ij++) {
// 	buf[_i*nao+_j] = tril[_ij];
// 	buf[_i+nao*_j] = tril[_ij]; // because going to use batched dgemm call on gpu
//       }
//   }
  
// #pragma omp parallel for
//   for (int i = 0; i < nij; i++) {
//     double * buf = &(_buf[i * nao * nao]);
    
//     const double D0 = 0;
//     const double D1 = 1;
//     const char SIDE_L = 'L';
//     const char UPLO_U = 'U';

//     double * _vout = vout + ij_pair*i;
    
//     dsymm_(&SIDE_L, &UPLO_U, &nao, &nao, &D1, buf, &nao, mo_coeff, &nao, &D0, _vout, &nao);    
//   }
  
}
#endif // end DEPRECATED Device::fdrv

/* ---------------------------------------------------------------------- */

/* SYCL kernels                                                           */

/* ---------------------------------------------------------------------- */

#if 1

void _getjk_rho(double * rho, double * dmtril, double * eri1, int nset, int naux, int nao_pair,
		double *cache)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

  // if(i >= nset) return;
  // if(j >= naux) return;
  
  int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
    item_ct1.get_local_id(0);
  int cache_id = item_ct1.get_local_id(0);
  
  // thread-local work
  
  const int indxi = i * nao_pair;
  const int indxj = j * nao_pair;
  
  double tmp = 0.0;
  
  if(i < nset && j < naux) { // only active threads do work
    
    while (k < nao_pair) {
      tmp += dmtril[indxi + k] * eri1[indxj + k];
      k += item_ct1.get_local_range(0); // * gridDim.z; // gridDim.z is just 1
    }
    
  }
  
  cache[cache_id] = tmp;
  
  // block

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // manually reduce values from threads within block

  int l = item_ct1.get_local_range(0) / 2;
  while (l != 0) {
    if(cache_id < l)
      cache[cache_id] += cache[cache_id + l];

    /*
    DPCT1118:0: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    l /= 2;
  }

  // store result in global array
  
  if(cache_id == 0)
    rho[i * naux + j] = cache[0];
}

#else

__global__ void _getjk_rho(double * rho, double * dmtril, double * eri1, int nset, int naux, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nset) return;
  if(j >= naux) return;

  double val = 0.0;
  for(int k=0; k<nao_pair; ++k) val += dmtril[i * nao_pair + k] * eri1[j * nao_pair + k];
  
  rho[i * naux + j] = val;
}

#endif

/* ---------------------------------------------------------------------- */

void _getjk_vj(double * vj, double * rho, double * eri1, int nset, int nao_pair, int naux, int chunk_size, int init)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
  const int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                item_ct1.get_local_id(0);

  int indxK = j * chunk_size + k;
  
  if(indxK >= nao_pair) return;

  double val = 0.0;
  for(int l=0; l<naux; ++l) val += rho[i * naux + l] * eri1[l * nao_pair + indxK];
  
  if(init) vj[i * nao_pair + indxK] = val;
  else vj[i * nao_pair + indxK] += val;
}

/* ---------------------------------------------------------------------- */

/* Interface functions calling SYCL kernels                               */

/* ---------------------------------------------------------------------- */

void DeviceJk::getjk_rho(double * rho, double * dmtril, double * eri, int nset, int naux, int nao_pair)
{
#if 1
  sycl::range<3> grid_size(1, naux, nset);
  sycl::range<3> block_size(_RHO_BLOCK_SIZE, 1, 1);
#else
  dim3 grid_size(nset, (naux + (_RHO_BLOCK_SIZE - 1)) / _RHO_BLOCK_SIZE, 1);
  dim3 block_size(1, _RHO_BLOCK_SIZE, 1);
#endif

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_rho :: nset= %i  naux= %i  nao_pair= %i  RHO_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nset, naux, nao_pair, _RHO_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  
#endif
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> cache_acc_ct1(
          sycl::range<1>(_RHO_BLOCK_SIZE), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grid_size * block_size, block_size),
          [=](sycl::nd_item<3> item_ct1) {
            _getjk_rho(
                rho, dmtril, eri, nset, naux, nao_pair,
                cache_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- get_jk::_getjk_rho :: finished\n");
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceJk::getjk_vj(double * vj, double * rho, double * eri, int nset, int nao_pair, int naux, int init)
{
  const int gs_nao_pair = (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE;
  const int chunk_size = (gs_nao_pair <= _SYCL_MAX_GRID_DIM_YZ) ? gs_nao_pair : _SYCL_MAX_GRID_DIM_YZ;
  const int num_chunks = (gs_nao_pair <= _SYCL_MAX_GRID_DIM_YZ) ? 1 : (gs_nao_pair / _SYCL_MAX_GRID_DIM_YZ + 1);
  
  sycl::range<3> grid_size(chunk_size, num_chunks, nset);
  sycl::range<3> block_size(_DOT_BLOCK_SIZE, 1, 1);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_vj :: nset= %i  nao_pair= %i  gs_nao_pair= %i  chunk_size= %i  num_chunks= %i  _DOT_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nset, nao_pair, gs_nao_pair, chunk_size, num_chunks, _DOT_BLOCK_SIZE, grid_size[0], grid_size[1], grid_size[2], block_size[0], block_size[1], block_size[2]);
#endif
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _getjk_vj(vj, rho, eri, nset, nao_pair, naux, chunk_size, init);
                    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- get_jk::_getjk_vj :: finished\n");
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

#endif
