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

#if 1

//https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

// modified to support nonsquare matrices

void _transpose(double * out, double * in, int nrow, int ncol,
                double cache[_TRANSPOSE_BLOCK_SIZE][_TRANSPOSE_BLOCK_SIZE+1])
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  
  int irow =
      item_ct1.get_group(2) * _TRANSPOSE_BLOCK_SIZE + item_ct1.get_local_id(2);
  int icol =
      item_ct1.get_group(1) * _TRANSPOSE_BLOCK_SIZE + item_ct1.get_local_id(1);

  // load tile into fast local memory

  const int indxi = irow * ncol + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < nrow && (icol+i) < ncol) // nonsquare
      cache[item_ct1.get_local_id(1) + i][item_ct1.get_local_id(2)] =
          in[indxi + i]; // threads read chunk of a row and write as a column
  }

  // block to ensure reads finish

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // swap indices

  irow =
      item_ct1.get_group(1) * _TRANSPOSE_BLOCK_SIZE + item_ct1.get_local_id(2);
  icol =
      item_ct1.get_group(2) * _TRANSPOSE_BLOCK_SIZE + item_ct1.get_local_id(1);

  // write tile to global memory

  const int indxo = irow * nrow + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < ncol && (icol + i) < nrow) // nonsquare
      out[indxo + i] =
          cache[item_ct1.get_local_id(2)][item_ct1.get_local_id(1) + i];
  }
}

#else

__global__ void _transpose(double * buf3, double * buf1, int nrow, int ncol)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= nrow) return;

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  while (j < ncol) {
    buf3[j*nrow + i] = buf1[i*ncol + j]; // these writes should be to SLM and then contiguous chunks written to global memory
    j += blockDim.y;
  }
  
}

#endif

/* ---------------------------------------------------------------------- */

void _vecadd(const double * in, double * out, int N)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if(i >= N) return;
    out[i] += in[i];
}

/* ---------------------------------------------------------------------- */

void _vecadd_batch(const double * in, double * out, int N, int num_batches)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if(i >= N) return;

    double val = 0.0;
    for(int j=0; j<num_batches; ++j) val += in[j*N + i];
    
    out[i] += val;
}

/* ---------------------------------------------------------------------- */

void _memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if(i >= N) return;
    
    for(int j=0; j<num_batches; ++j) inout[j*stride + offset + i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void _set_to_zero(double * array, int size)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i>=size) return; 
    array[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

SYCL_EXTERNAL void _veccopy(const double * src, double *dest, int size)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= size) return;
    dest[i] = src[i];
}

/* ---------------------------------------------------------------------- */

void _transpose_021(double * in, double * out, int ax1, int ax2, int ax3)
{
    // abc->acb
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

    if(i >= ax1) return;
    if(j >= ax2) return;
    if(k >= ax3) return;

    int inputIndex = (i*ax3+k)*ax2+j;
    int outputIndex = (i*ax2+j)*ax3+k;
    //printf("%i %i %i %f\n",i, j, k, in[inputIndex]);
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

void _transpose_102(double * in, double * out, int ax1, int ax2, int ax3)
{
    // abc->bac
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

    if(i >= ax1) return;
    if(j >= ax2) return;
    if(k >= ax3) return;

    int inputIndex = (j*ax1+i)*ax3+k;
    int outputIndex = (i*ax2+j)*ax3+k;
    //printf("%i %i %i %f\n",i, j, k, in[inputIndex]);
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

void _transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4)
{
    // rs(bazl)k->(bazl)skr
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx1 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    int idx2 = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
               item_ct1.get_local_id(1);
    int idx3 = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
               item_ct1.get_local_id(0);

    if(idx1 >= ax1) return;
    if(idx2 >= ax2) return;
    if(idx3 >= ax3) return;
    int inputIndex, outputIndex;
    for (int idx4=0;idx4<ax4;++idx4){
      outputIndex = ((idx3*ax2 + idx2)*ax4 + idx4)*ax1 + idx1;
      inputIndex = ((idx1*ax2 + idx2)*ax3 + idx3)*ax4 + idx4;
      //printf("input: %i: %f output: %i\n",inputIndex, in[inputIndex], outputIndex);
      out[outputIndex] = in[inputIndex];}
}

/* ---------------------------------------------------------------------- */

void _transpose_3210(double* in, double* out, int nmo, int ncas)
{
    //a.transpose(3,2,1,0)-ncas,ncas,ncas,nmo
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);

    if(i >= ncas) return;
    if(j >= ncas) return;
    if(k >= ncas) return;
    
    for(int l=0;l<nmo;++l){
      int inputIndex = ((i*ncas+j)*ncas+k)*nmo+l;
      int outputIndex = l*ncas*ncas*ncas+k*ncas*ncas+j*ncas+i;
      out[outputIndex]=in[inputIndex];
    }
}

/* ---------------------------------------------------------------------- */

void _transpose_120(double * in, double * out, int naux, int nao, int ncas)
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
    int outputIndex = j*nao*naux  + k*naux + i;
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

#if 1

void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
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
  for(int k=0; k<nao; ++k) buf[indx+k] = tril[ map[indx+k] ];  
}

#else

__global__ void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao*nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  buf[j] = tril[ map[j] ];
}

#endif

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose(double * out, double * in, int nrow, int ncol)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- calling _transpose()\n");
#endif
            
#if 1 
  sycl::range<3> grid_size(1, _TILE(ncol, _TRANSPOSE_BLOCK_SIZE),
			   _TILE(nrow, _TRANSPOSE_BLOCK_SIZE));
  sycl::range<3> block_size(1, _TRANSPOSE_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE);
#else
  dim3 grid_size(nrow, 1, 1);
  dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
#endif

  sycl::queue * s = ctx.pm->dev_get_queue();

  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<
          double[_TRANSPOSE_BLOCK_SIZE][_TRANSPOSE_BLOCK_SIZE+1],
          0>
          cache_acc_ct1(cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                       [=](sycl::nd_item<3> item_ct1) {
                         _transpose(out, in, nrow, ncol, cache_acc_ct1);
                       });
    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- transpose :: nrow= %i  ncol= %i _TRANSPOSE_BLOCK_SIZE= %i  _TRANSPOSE_NUM_ROWS= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 nrow, ncol, _TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::vecadd(const double * in, double * out, int N)
{
  sycl::range<3> block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, 1, _TILE(N, block_size[2]));
                      
  sycl::queue * s = ctx.pm->dev_get_queue();
                    
  /*
  DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */     
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _vecadd(in, out, N);
                    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- general::vecadd :: N= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
         N, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::vecadd_batch(const double * in, double * out, int N, int num_batches)
{
  sycl::range<3> block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, 1, _TILE(N, block_size[2]));

  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:14: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _vecadd_batch(in, out, N, num_batches);
                    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- general::vecadd_batch :: N= %i  num_batches= %i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 N, num_batches, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
 
  sycl::range<3> block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, 1, _TILE(N, block_size[2]));

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- Inside general::memset_zero_batch_stride :: stride= %i  offset= %i  N= %i  num_batches= %i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 stride, offset, N, num_batches, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
  
  sycl::queue * s = ctx.pm->dev_get_queue();
 
  /*
  DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _memset_zero_batch_stride(inout, stride, offset, N,
                                                num_batches);
                    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- Leaving general::memset_zero_batch_stride :: stride= %i  offset= %i  N= %i  num_batches= %i  grid_size= %lu %lu %lu  block_size= %lu %lu %lu\n",
	 stride, offset, N, num_batches, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::set_to_zero(double * array, int size)
{
  sycl::queue * s = ctx.pm->dev_get_queue();
  
#if 1
  sycl::range<3> block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, 1, _TILE(size, block_size[2]));
  /*
    DPCT1049:35: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});
    
    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _set_to_zero(array, size);
                    });
  }
  ctx.pm->dev_check_errors();
#else
  cudaMemSet(array,0, size*sizeof(double), s); //Is this better?
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::veccopy(const double * src, double *dest, int size)
{
  sycl::queue * s = ctx.pm->dev_get_queue();
  sycl::range<3> block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> grid_size(1, 1, _TILE(size, block_size[2]));
  /*
  DPCT1049:40: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _veccopy(src, dest, size);
                    });
  }
  ctx.pm->dev_check_errors();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_021( double * in, double * out, int ax1, int ax2, int ax3)
{
  sycl::queue * s = ctx.pm->dev_get_queue();
  
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> block_size(1, 1, 1);
  sycl::range<3> grid_size(_TILE(ax3, block_size[0]), _TILE(ax2, block_size[1]),
                       _TILE(ax1, block_size[2]));

  /*
  DPCT1049:40: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_021(in, out, ax1, ax2, ax3);
                    });
  }

  ctx.pm->dev_check_errors();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_102( double * in, double * out, int ax1, int ax2, int ax3)
{
  sycl::queue * s = ctx.pm->dev_get_queue();
  
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  sycl::range<3> block_size(1, 1, 1);
  sycl::range<3> grid_size(_TILE(ax3, block_size[0]), _TILE(ax2, block_size[1]),
                       _TILE(ax1, block_size[2]));

  /*
  DPCT1049:40: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_102(in, out, ax1, ax2, ax3);
                    });
  }

  ctx.pm->dev_check_errors();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4)
{
  sycl::queue * s = ctx.pm->dev_get_queue();
  
  sycl::range<3> block_size(1, 1, 1);
  sycl::range<3> grid_size(_TILE(ax3, block_size[0]), _TILE(ax2, block_size[1]),
                       _TILE(ax1, block_size[2]));
  {
    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_2130(in, out, ax1, ax2, ax3, ax4);
                    });
  }
  ctx.pm->dev_check_errors();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_3210(double* in, double* out, int nmo, int ncas)
{
  sycl::range<3> block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_size(_TILE(ncas, block_size[0]),
			   _TILE(ncas, block_size[1]),
			   _TILE(ncas, block_size[2]));
  
  sycl::queue * s = ctx.pm->dev_get_queue();

  /*
  DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    //    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_3210(in, out, nmo, ncas);
                    });
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::transpose_3210 :: ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 ncas, _DEFAULT_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_120(double * in, double * out, int naux, int nao, int ncas, int order)
{
  sycl::queue * s = ctx.pm->dev_get_queue();

  int na = nao;
  int nb = ncas;
  
  if(order == 1) {
    na = ncas;
    nb = nao;
  }

  sycl::range<3> block_size(1, 1, 1);
  sycl::range<3> grid_size(nb, na, _TILE(naux, block_size[2])); // originally nmo, nmo
  
  {
    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _transpose_120(in, out, naux, nao, ncas);
                    });
  }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::getjk_unpack_buf2(double * buf2, double * eri, int * map, int naux, int nao, int nao_pair)
{
#if 1
  sycl::range<3> grid_size(1, _TILE(nao, _UNPACK_BLOCK_SIZE), naux);
  sycl::range<3> block_size(1, _UNPACK_BLOCK_SIZE, 1);
#else
  dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif

  sycl::queue * s = ctx.pm->dev_get_queue();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_unpack_buf2 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %zu %zu %zu  block_size= %zu %zu %zu\n",
	 naux, nao, _UNPACK_BLOCK_SIZE, grid_size[0],grid_size[1],grid_size[2],block_size[0],block_size[1],block_size[2]);
#endif

  {
    //dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

    s->parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                    [=](sycl::nd_item<3> item_ct1) {
                      _getjk_unpack_buf2(buf2, eri, map, naux, nao, nao_pair);
                    });
  }

#ifdef _DEBUG_DEVICE
  ctx.pm->dev_stream_wait();
  printf("LIBGPU ::  -- get_jk::_getjk_vj :: finished\n");
  ctx.pm->dev_check_errors();
#endif
}


#endif
