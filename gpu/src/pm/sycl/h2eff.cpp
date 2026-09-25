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

void _extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

  if(i >= ncas) return;
  if(j >= ncas) return;
  
  small_mat[i * ncas + j] = big_mat[(i + ncore) * nmo + (j + ncore)];
}

/* ---------------------------------------------------------------------- */

void _transpose_2310(double * in, double * out, int nmo, int ncas)
{
    //a.transpose(2,3,1,0)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

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

void _transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
    //Pum->muP
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

    if(i >= naux) return;
    if(j >= ncas) return;
    if(k >= nao) return;

    int inputIndex = i*nao*ncas+j*nao+k;
    int outputIndex = k*ncas*naux  + j*naux + i;
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

void _pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
  const int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                item_ct1.get_local_id(0);

  if(i >= nmo) return;
  if(j >= ncas) return;
  if(k >= ncas_pair) return;

  double * out_buf = &(out[(i*ncas + j) * ncas_pair]);
  double * in_buf = &(in[(i*ncas + j) * ncas*ncas]);

  out_buf[ k ] = in_buf[ map[k] ];
}

/* ---------------------------------------------------------------------- */

void _unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

  if(i >= nmo*ncas) return;
  if(j >= ncas*ncas) return;

  double * in_buf = &(in[i * ncas_pair ]);
  double * out_buf = &(out[i * ncas*ncas ]);

  out_buf[j] = in_buf[ map[j] ];
}

/* ---------------------------------------------------------------------- */

void _pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]=in[j*ncas*nmo + i];
}

/* ---------------------------------------------------------------------- */

void _pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]+=in[j*ncas*nmo + i];
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(1, _TILE(ncas, block_size[1]), _TILE(ncas, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _extract_submatrix(big_mat, small_mat, ncas, ncore, nmo);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- extract_submatrix :: ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 ncas, _UNPACK_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(1,
			   _TILE(ncas * ncas, _UNPACK_BLOCK_SIZE),
			   _TILE(nmo * ncas, _UNPACK_BLOCK_SIZE));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    // dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
    //                              {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
		    [=](sycl::nd_item<3> item_ct1) {
		      _unpack_h2eff_2d(in, out, map, nmo, ncas, ncas_pair);
		    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- _unpack_h2eff_2d :: nmo*ncas= %i  ncas*ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nmo*ncas, ncas*ncas, _UNPACK_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::transpose_2310(double * in, double * out, int nmo, int ncas)
{
  sycl::range<3> block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_size(_TILE(ncas, block_size[0]),
			   _TILE(ncas, block_size[1]),
			   _TILE(nmo, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_2310(in, out, nmo, ncas);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::transpose_2310 :: nmo= %i  ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nmo, ncas, _DEFAULT_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
  sycl::range<3> block_size(_UNPACK_BLOCK_SIZE, 1, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(_TILE(nao, block_size[0]),
			   _TILE(ncas, block_size[1]),
			   _TILE(naux, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::transpose_210 :: naux= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 naux, ncas, _UNPACK_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
  
  /*
  DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_210(in, out, naux, nao, ncas);
                    });
  }
  
#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- h2eff_df_contract1::transpose_210 :: finished\n");
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  sycl::range<3> block_size(_UNPACK_BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_size(_TILE(ncas_pair, _DEFAULT_BLOCK_SIZE), ncas, nmo);
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _pack_h2eff_2d(in, out, map, nmo, ncas, ncas_pair);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::_pack_h2eff_2d :: nmo= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nmo, ncas, _UNPACK_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(1,
			   _TILE(ncas * ncas, block_size[1]),
			   _TILE(nmo * ncas, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nmo*ncas, ncas*ncas, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
  
  /*
  DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _pack_d_vuwM(in, out, map, nmo, ncas, ncas_pair);
                    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vuwM :: finished\n");
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  sycl::range<3> grid_size(1, _TILE(ncas * ncas, block_size[1]),
			   _TILE(nmo * ncas, block_size[2]));

  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _pack_d_vuwM_add(in, out, map, nmo, ncas, ncas_pair);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM_add :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %zu %zu %zu block_size= %zu %zu %zu\n",
	 nmo*ncas, ncas*ncas, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}


#endif
