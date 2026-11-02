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

void _get_bufd( const double* bufpp, double* bufd, int naux, int nmo)
{

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < naux && j < nmo) {
        bufd[i * nmo + j] = bufpp[(i*nmo + j)*nmo + j];
    }
}

/* ---------------------------------------------------------------------- */

void _get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao, int nmo)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < ncas && j < nao) {
        small_mat[i * nao + j] = big_mat[j*nmo + i+ncore];
    }
}


/* ---------------------------------------------------------------------- */

void _get_bufpa (const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
  const int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                item_ct1.get_local_id(0);

  if(i >= naux) return;
  if(j >= nmo) return;
  if(k >= ncas) return;
  
  int inputIndex = (i*nmo + j)*nmo + k+ncore;
  int outputIndex = (i*nmo + j)*ncas + k;
  bufpa[outputIndex] = bufpp[inputIndex];
}

/* ---------------------------------------------------------------------- */

void _get_bufaa (const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
  const int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                item_ct1.get_local_id(0);

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
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(ncas, _TILE(nmo, block_size[1]), _TILE(naux, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _get_bufpa(bufpp, bufpa, naux, nmo, ncore, ncas);
                    });
  }
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufaa(const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
  sycl::range<3> block_size(1, 1, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(ncas, ncas, _TILE(naux, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _get_bufaa(bufpp, bufaa, naux, nmo, ncore, ncas);
                    });
  }
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao, int nmo)
{
  sycl::range<3> block_size(1, 1, 1);
  sycl::range<3> grid_size(1, _TILE(nao, block_size[1]), _TILE(ncas, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _get_mo_cas(big_mat, small_mat, ncas, ncore, nao, nmo);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::_get_mo_cas :: ncas= %i  nao= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 ncas, nao, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufd( const double* bufpp, double* bufd, int naux, int nmo)
{
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(1, _TILE(nmo, block_size[1]), _TILE(naux, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _get_bufd(bufpp, bufd, naux, nmo);
                    });
  }
}

/* ---------------------------------------------------------------------- */

#endif