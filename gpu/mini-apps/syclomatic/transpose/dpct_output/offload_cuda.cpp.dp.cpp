#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>

#define _TRANSPOSE_BLOCK_SIZE 16
#define _TRANSPOSE_NUM_ROWS 16

#define _TILE(A,B) (A + B - 1) / B

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

// ----------------------------------------------------------------
// GPU Kernels
// ----------------------------------------------------------------

//https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
// modified to support nonsquare matrices

void _transpose_gpu_v1(real_t * out, real_t * in, const int nrow, const int ncol,
                       const sycl::nd_item<3> &item_ct1,
                       sycl::local_accessor<real_t, 2> cache)
{

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

// ----------------------------------------------------------------

void _transpose_gpu_v2(real_t * out, real_t * in, const int nrow, const int ncol,
                       const sycl::nd_item<3> &item_ct1,
                       sycl::local_accessor<real_t, 2> cache)
{

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

// ----------------------------------------------------------------

void _transpose_naive_gpu(real_t * out, real_t * in, const int nrow, const int ncol,
                          const sycl::nd_item<3> &item_ct1)
{
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

  if(i >= nrow) return;

  int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);

  while (j < ncol) {
    out[j*nrow + i] = in[i*ncol + j];
    j += item_ct1.get_local_range(1);
  }
  
}

// ----------------------------------------------------------------

void _copy_naive_gpu(real_t * out, real_t * in, const int nrow, const int ncol,
                     const sycl::nd_item<3> &item_ct1)
{
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

  if(i >= nrow) return;

  int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);

  while (j < ncol) {
    out[i*ncol + j] = in[i*ncol + j];
    j += item_ct1.get_local_range(1);
  }
  
}

// ----------------------------------------------------------------
// Host-side functions
// ----------------------------------------------------------------

void copy_naive_gpu(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  sycl::range<3> grid_size(1, 1, num_rows);
  sycl::range<3> block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);

  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(grid_size * block_size, block_size),
      [=](sycl::nd_item<3> item_ct1) {
        _copy_naive_gpu(b, a, num_rows, num_cols, item_ct1);
      });
}

// ----------------------------------------------------------------

void transpose_naive_gpu(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  sycl::range<3> grid_size(1, 1, num_rows);
  sycl::range<3> block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);

  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(grid_size * block_size, block_size),
      [=](sycl::nd_item<3> item_ct1) {
        _transpose_naive_gpu(b, a, num_rows, num_cols, item_ct1);
      });
}

// ----------------------------------------------------------------

void transpose_gpu_v1(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  sycl::range<3> grid_size(1, _TILE(num_cols, _TRANSPOSE_BLOCK_SIZE),
                           _TILE(num_rows, _TRANSPOSE_BLOCK_SIZE));
  sycl::range<3> block_size(1, _TRANSPOSE_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE);

  //  printf("\ngrid_size= %i %i %i\n", _TILE(_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE), _TILE(_NUM_COLS, _TRANSPOSE_BLOCK_SIZE), 1);
  //  printf("block_size= %i %i %i\n",_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:0: '_TRANSPOSE_BLOCK_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    /*
    DPCT1101:1: '_TRANSPOSE_BLOCK_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 2> cache_acc_ct1(
        sycl::range<2>(16 /*_TRANSPOSE_BLOCK_SIZE*/,
                       16 /*_TRANSPOSE_BLOCK_SIZE*/),
        cgh);

    cgh.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](sycl::nd_item<3> item_ct1) {
                       _transpose_gpu_v1(b, a, num_rows, num_cols, item_ct1,
                                         cache_acc_ct1);
                     });
  });
}

// ----------------------------------------------------------------

void transpose_gpu_v2(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  sycl::range<3> grid_size(1, _TILE(num_cols, _TRANSPOSE_BLOCK_SIZE),
                           _TILE(num_rows, _TRANSPOSE_BLOCK_SIZE));
  sycl::range<3> block_size(1, _TRANSPOSE_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE);

  //  printf("\ngrid_size= %i %i %i\n", _TILE(_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE), _TILE(_NUM_COLS, _TRANSPOSE_BLOCK_SIZE), 1);
  //  printf("block_size= %i %i %i\n",_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:2: '_TRANSPOSE_BLOCK_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    /*
    DPCT1101:3: '_TRANSPOSE_BLOCK_SIZE+1' expression was replaced with a
    value. Modify the code to use the original expression, provided in
    comments, if it is correct.
    */
    sycl::local_accessor<real_t, 2> cache_acc_ct1(
        sycl::range<2>(16 /*_TRANSPOSE_BLOCK_SIZE*/,
                       17 /*_TRANSPOSE_BLOCK_SIZE+1*/),
        cgh);

    cgh.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](sycl::nd_item<3> item_ct1) {
                       _transpose_gpu_v2(b, a, num_rows, num_cols, item_ct1,
                                         cache_acc_ct1);
                     });
  });
}

// ----------------------------------------------------------------
