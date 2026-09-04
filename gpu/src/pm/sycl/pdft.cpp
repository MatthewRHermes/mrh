/* -*- c++ -*- */

#if defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)

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

//#define _DEBUG_DEVICE
//#define _DEBUG_H2EFF
//#define _DEBUG_H2EFF2
//#define _DEBUG_H2EFF_DF
//#define _DEBUG_AO2MO

#define _TILE(A,B) (A + B - 1) / B
#define _SYCL_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void _get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if(i >= ngrid) return;

    Pi[i] += rho[i] * rho[i];
}

/* ---------------------------------------------------------------------- */

void _make_gridkern(double * mo_grid, double * gridkern, int ngrid, int ncas)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

    if(i >= ngrid) return;
    if(j >= ncas) return;
    if(k >= ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_mo_grid = &(mo_grid[i*ncas]);
    tmp_gridkern[j*ncas+k] = tmp_mo_grid[j]*tmp_mo_grid[k];
}

/* ---------------------------------------------------------------------- */

void _make_buf_pdft(double * gridkern, double * cascm2, double * out, int ngrid, int ncas)
{
    //TODO: convert this to a dgemm
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

    if(i >= ngrid) return;
    if(j >= ncas*ncas) return;
    if(k >= ncas*ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_cascm2 = &(cascm2[j*ncas*ncas]);
    double * tmp_out = &(out[i*ncas*ncas+j]);
    tmp_out[0] += tmp_gridkern[k]*tmp_cascm2[k];
}

/* ---------------------------------------------------------------------- */

void _make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
    //TODO: convert this to a dgemm
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

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
  sycl::range<3> block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, 1, _TILE(ngrid, block_size[2]));

  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _get_rho_to_Pi(rho, Pi, ngrid);
                    });
  }
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rho_to_Pi :: N= %i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 ngrid, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_gridkern(double * d_mo_grid, double * d_gridkern, int ngrid, int ncas)
{
  sycl::range<3> block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE,
              _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(_TILE(ncas, block_size[0]),
			   _TILE(ncas, block_size[1]),
			   _TILE(ngrid, block_size[2]));

  sycl::queue * s = ctx.pm->dev_get_queue();
  
  /*
  DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _make_gridkern(d_mo_grid, d_gridkern, ngrid, ncas);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_gridkern :: N= %i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 ncas, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_buf_pdft(double * gridkern, double * buf, double * cascm2, int ngrid, int ncas)
{
  sycl::range<3> block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE,
                        _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(_TILE(ncas * ncas, block_size[0]),
			   _TILE(ncas * ncas, block_size[1]),
			   _TILE(ngrid, block_size[2]));

  sycl::queue * s = ctx.pm->dev_get_queue();
  
  // buf = aij, klij ->akl, gridkern, cascm2
  /*
  DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //  dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _make_buf_pdft(gridkern, cascm2, buf, ngrid, ncas);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_gridkern :: N= %i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 ncas, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif

}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
  sycl::range<3> block_size(1, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, _TILE(ncas * ncas, block_size[1]),
			   _TILE(ngrid, block_size[2]));

  sycl::queue * s = ctx.pm->dev_get_queue();
  
  /*
  DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _make_Pi_final(gridkern, buf, Pi, ngrid, ncas);
                    });
  }
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_Pi_final; :: Ngrid= %i Ncas =%i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 ngrid, ncas, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}


#endif
