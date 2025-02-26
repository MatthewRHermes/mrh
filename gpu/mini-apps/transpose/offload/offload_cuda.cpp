#if defined(_GPU_CUDA)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include "../pm/pm.h"

#define _TRANSPOSE_BLOCK_SIZE 16
#define _TRANSPOSE_NUM_ROWS 16

#define _TILE(A,B) (A + B - 1) / B

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

using namespace PM_NS;

class PM * pm_ = nullptr;

// ----------------------------------------------------------------
// GPU Kernels
// ----------------------------------------------------------------

//https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
// modified to support nonsquare matrices

__global__ void _transpose_gpu_v1(real_t * out, real_t * in, const int nrow, const int ncol)
{
  __shared__ real_t cache[_TRANSPOSE_BLOCK_SIZE][_TRANSPOSE_BLOCK_SIZE];
  
  int irow = blockIdx.x * _TRANSPOSE_BLOCK_SIZE + threadIdx.x;
  int icol = blockIdx.y * _TRANSPOSE_BLOCK_SIZE + threadIdx.y;

  // load tile into fast local memory

  const int indxi = irow * ncol + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < nrow && (icol+i) < ncol) // nonsquare
      cache[threadIdx.y + i][threadIdx.x] = in[indxi + i]; // threads read chunk of a row and write as a column
  }

  // block to ensure reads finish
  
  __syncthreads();

  // swap indices
  
  irow = blockIdx.y * _TRANSPOSE_BLOCK_SIZE + threadIdx.x;
  icol = blockIdx.x * _TRANSPOSE_BLOCK_SIZE + threadIdx.y;

  // write tile to global memory

  const int indxo = irow * nrow + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < ncol && (icol + i) < nrow) // nonsquare
      out[indxo + i] = cache[threadIdx.x][threadIdx.y + i];
  }
}

// ----------------------------------------------------------------

__global__ void _transpose_gpu_v2(real_t * out, real_t * in, const int nrow, const int ncol)
{
  __shared__ real_t cache[_TRANSPOSE_BLOCK_SIZE][_TRANSPOSE_BLOCK_SIZE+1];
  
  int irow = blockIdx.x * _TRANSPOSE_BLOCK_SIZE + threadIdx.x;
  int icol = blockIdx.y * _TRANSPOSE_BLOCK_SIZE + threadIdx.y;

  // load tile into fast local memory

  const int indxi = irow * ncol + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < nrow && (icol+i) < ncol) // nonsquare
      cache[threadIdx.y + i][threadIdx.x] = in[indxi + i]; // threads read chunk of a row and write as a column
  }

  // block to ensure reads finish
  
  __syncthreads();

  // swap indices
  
  irow = blockIdx.y * _TRANSPOSE_BLOCK_SIZE + threadIdx.x;
  icol = blockIdx.x * _TRANSPOSE_BLOCK_SIZE + threadIdx.y;

  // write tile to global memory

  const int indxo = irow * nrow + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < ncol && (icol + i) < nrow) // nonsquare
      out[indxo + i] = cache[threadIdx.x][threadIdx.y + i];
  }
}

// ----------------------------------------------------------------

__global__ void _transpose_naive_gpu(real_t * out, real_t * in, const int nrow, const int ncol)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= nrow) return;

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  while (j < ncol) {
    out[j*nrow + i] = in[i*ncol + j];
    j += blockDim.y;
  }
  
}

// ----------------------------------------------------------------

__global__ void _copy_naive_gpu(real_t * out, real_t * in, const int nrow, const int ncol)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= nrow) return;

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  while (j < ncol) {
    out[i*ncol + j] = in[i*ncol + j];
    j += blockDim.y;
  }
  
}

// ----------------------------------------------------------------
// Host-side functions
// ----------------------------------------------------------------

void init_pm(class PM * pm)
{
  pm_ = pm;
}

void copy_naive_gpu(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  dim3 grid_size(num_rows, 1, 1);
  dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
  
  _copy_naive_gpu<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
}

// ----------------------------------------------------------------

void transpose_naive_gpu(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  dim3 grid_size(num_rows, 1, 1);
  dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
  
  _transpose_naive_gpu<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
}

// ----------------------------------------------------------------

void transpose_gpu_v1(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  dim3 grid_size(_TILE(num_rows, _TRANSPOSE_BLOCK_SIZE), _TILE(num_cols, _TRANSPOSE_BLOCK_SIZE), 1);
  dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
  //  printf("\ngrid_size= %i %i %i\n", _TILE(_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE), _TILE(_NUM_COLS, _TRANSPOSE_BLOCK_SIZE), 1);
  //  printf("block_size= %i %i %i\n",_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
  _transpose_gpu_v1<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
}

// ----------------------------------------------------------------

void transpose_gpu_v2(real_t * b, real_t * a, const int num_rows, const int num_cols)
{
  dim3 grid_size(_TILE(num_rows, _TRANSPOSE_BLOCK_SIZE), _TILE(num_cols, _TRANSPOSE_BLOCK_SIZE), 1);
  dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
  //  printf("\ngrid_size= %i %i %i\n", _TILE(_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE), _TILE(_NUM_COLS, _TRANSPOSE_BLOCK_SIZE), 1);
  //  printf("block_size= %i %i %i\n",_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
  _transpose_gpu_v2<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
}

// ----------------------------------------------------------------

#endif
