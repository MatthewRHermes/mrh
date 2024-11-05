#if defined(_USE_CPU)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include "pm.h"

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
// Host-side functions
// ----------------------------------------------------------------

void init_pm(class PM * pm)
{
  pm_ = pm;
}

// void copy_naive_gpu(real_t * b, real_t * a, const int num_rows, const int num_cols)
// {
//   dim3 grid_size(num_rows, 1, 1);
//   dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
  
//   _copy_naive_gpu<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
// }

// // ----------------------------------------------------------------

// void transpose_naive_gpu(real_t * b, real_t * a, const int num_rows, const int num_cols)
// {
//   dim3 grid_size(num_rows, 1, 1);
//   dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
  
//   _transpose_naive_gpu<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
// }

// // ----------------------------------------------------------------

// void transpose_gpu_v1(real_t * b, real_t * a, const int num_rows, const int num_cols)
// {
//   dim3 grid_size(_TILE(num_rows, _TRANSPOSE_BLOCK_SIZE), _TILE(num_cols, _TRANSPOSE_BLOCK_SIZE), 1);
//   dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
//   //  printf("\ngrid_size= %i %i %i\n", _TILE(_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE), _TILE(_NUM_COLS, _TRANSPOSE_BLOCK_SIZE), 1);
//   //  printf("block_size= %i %i %i\n",_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
//   _transpose_gpu_v1<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
// }

// // ----------------------------------------------------------------

// void transpose_gpu_v2(real_t * b, real_t * a, const int num_rows, const int num_cols)
// {
//   dim3 grid_size(_TILE(num_rows, _TRANSPOSE_BLOCK_SIZE), _TILE(num_cols, _TRANSPOSE_BLOCK_SIZE), 1);
//   dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
//   //  printf("\ngrid_size= %i %i %i\n", _TILE(_NUM_ROWS, _TRANSPOSE_BLOCK_SIZE), _TILE(_NUM_COLS, _TRANSPOSE_BLOCK_SIZE), 1);
//   //  printf("block_size= %i %i %i\n",_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
  
//   _transpose_gpu_v2<<<grid_size, block_size>>>(b, a, num_rows, num_cols);
// }

// // ----------------------------------------------------------------

#endif
