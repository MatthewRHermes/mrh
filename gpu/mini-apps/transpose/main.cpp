#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include <mpi.h>
#include <omp.h>

#include "pm/pm.h"

// polymer_async : 6-31g
// -- nfrags = 16 :: nrow= 85,440  ncol= 356 copy_naive= 426 GB/s  transpose_naive= 151 GB/s  transpose_gpu= 383 GB/s  transpose_gpu_2= 538 GB/s
// --                nrow= 50,552  ncol= 356 copy_naive= 390 GB/s  transpose_naive= 126 GB/s  transpose_gpu= 273 GB/s  transpose_gpu_2= 396 GB/s
// --                nrow= 10,560  ncol=  44 copy_naive= 198 GB/s  transpose_naive= 158 GB/s  transpose_gpu= 200 GB/s  transpose_gpu_2= 263 GB/s
// --                nrow=  6,248  ncol=  44 copy_naive= 148 GB/s  transpose_naive= 132 GB/s  transpose_gpu= 191 GB/s  transpose_gpu_2= 249 GB/s

// fefefe_15_16
// --   naux=240, nao=708   nrow= 169,920 ncol= 708
// --                                        copy_naive= 449 GB/s  transpose_naive= 153 GB/s  transpose_gpu= 385 GB/s  transpose_gpu_2= 525 GB/s
// --   naux=169, nao=708   nrow= 119,652 ncol= 708
// --                                        copy_naive= 442 GB/s  transpose_naive= 159 GB/s  transpose_gpu= 501 GB/s  transpose_gpu_2= 533 GB/

#define _NUM_ROWS 50552
#define _NUM_COLS 356

#define _TOL 1e-8
#define _NUM_ITERATIONS_CPU 20
#define _NUM_ITERATIONS_GPU 1000

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

using namespace PM_NS;

extern void init_pm(class PM *);
extern void transpose(real_t *, real_t *, const int, const int);
extern void copy_naive_gpu(real_t *, real_t *, const int, const int);
extern void transpose_naive_gpu(real_t *, real_t *, const int, const int);
extern void transpose_gpu_v1(real_t *, real_t *, const int, const int);
extern void transpose_gpu_v2(real_t *, real_t *, const int, const int);
extern void transpose_gpu_v3(real_t *, real_t *, const int, const int);

// ----------------------------------------------------------------

void transpose_naive_cpu(real_t * out, real_t * in, int num_rows, int num_cols)
{
  for(int i=0; i<num_rows; ++i)
    for(int j=0; j<num_cols; ++j)
      out[j*num_rows + i] = in[i*num_cols + j];
}

// ----------------------------------------------------------------

int check_result(real_t * ref, real_t * test, int n, const char * name)
{
  int err = 0;
  for(int i=0; i<n; ++i) {
    real_t diff = (ref[i] - test[i]) * (ref[i] - test[i]);
    if(diff > _TOL) err++;
  }
  
  if(err == 0) printf("Results from %s are correct!! :) \n", name);
  else printf("Results from %s are incorrect!! :( \n", name);
  
  return err;
}

// ----------------------------------------------------------------

void print_matrix(real_t * data, int num_rows, int num_cols, const char * name)
{
  printf("\nMatrix[%s] : %i x %i \n",name, num_rows, num_cols);
  for(int i=0; i<num_rows; ++i) {
    for(int j=0; j<num_cols; ++j) printf(" %f", data[i*num_cols + j]);
    printf("\n");
  }
}

// ----------------------------------------------------------------

void print_summary(double t, int num_rows, int num_cols, int num_iter, const char * name)
{
  double size = num_rows * num_cols * sizeof(real_t) / 1024.0 / 1024.0 / 1024.0; // GB

  double bandwidth = size * num_iter / t; // GB/s

  printf("\nMatrix[%s] : %i x %i  num_iter= %i  size= %f  [MB]  time= %f  [ms]  bandwidth= %f  [GB/s]\n",
	 name, num_rows, num_cols, num_iter, size, t*1000.0, bandwidth);
}

// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  int me,nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

#ifdef _SINGLE_PRECISION
  if(me == 0) printf("Using single-precision\n\n");
#else
  if(me == 0) printf("Using double-precision\n\n");
#endif
  
  // ----------------------------------------------------------------

  class PM * pm = new PM();

  init_pm(pm);
  
  int num_devices = pm->dev_num_devices();

  if(me == 0) {
    printf("# of devices= %i\n",num_devices);
    pm->dev_properties(num_devices);
  }
  
  // Device ID

  int device_id = me % num_devices;

  pm->dev_set_device(device_id);
  
  for(int i=0; i<nranks; ++i) {
    if(i == me) {
      printf("Rank %i running on GPU %i!\n",me,device_id);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  real_t * a = (real_t*) malloc(_NUM_ROWS * _NUM_COLS * sizeof(real_t));
  real_t * b = (real_t*) malloc(_NUM_ROWS * _NUM_COLS * sizeof(real_t));
  real_t * r = (real_t*) malloc(_NUM_ROWS * _NUM_COLS * sizeof(real_t));

  // Initialize host
  
  for(int i=0; i<_NUM_ROWS; ++i) {
    for(int j=0; j<_NUM_COLS; ++j) {
      a[i*_NUM_COLS + j] = i * _NUM_COLS + j;
      r[j*_NUM_ROWS + i] = i * _NUM_COLS + j; // transposed reference
    }
  }

  // ----------------------------------------------------------------
  
  transpose_naive_cpu(b, a, _NUM_ROWS, _NUM_COLS);
  double t0 = MPI_Wtime();
  for(int i=0; i<_NUM_ITERATIONS_CPU; ++i)
    transpose_naive_cpu(b, a, _NUM_ROWS, _NUM_COLS);
  double t = MPI_Wtime() - t0;

  print_summary(t, _NUM_ROWS, _NUM_COLS, _NUM_ITERATIONS_CPU, "transpose_naive_cpu");

  check_result(r, b, _NUM_ROWS*_NUM_COLS, "naive_cpu");
  
  //  print_matrix(a, _NUM_ROWS, _NUM_COLS, "Original a");

  //  print_matrix(b, _NUM_COLS, _NUM_ROWS, "Transposed b");

  //  print_matrix(r, _NUM_COLS, _NUM_ROWS, "Reference r");

  // Create device buffers and transfer data to device
  
  real_t * d_a = (real_t *) pm->dev_malloc(_NUM_ROWS * _NUM_COLS * sizeof(real_t));
  real_t * d_b = (real_t *) pm->dev_malloc(_NUM_ROWS * _NUM_COLS * sizeof(real_t));
  
  for(int i=0; i<_NUM_ROWS*_NUM_COLS; ++i) b[i] = -1.0;
 
  pm->dev_push(d_a, a, _NUM_ROWS * _NUM_COLS * sizeof(real_t));

  // ----------------------------------------------------------------
  
  // Execute kernel

  copy_naive_gpu(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  
  t0 = MPI_Wtime();
  for(int i=0; i<_NUM_ITERATIONS_GPU; ++i)
    copy_naive_gpu(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  t = MPI_Wtime() - t0;

  print_summary(t, _NUM_ROWS, _NUM_COLS, _NUM_ITERATIONS_GPU, "_copy_naive_gpu");
  
  // ----------------------------------------------------------------
  
  // Execute kernel

  transpose_naive_gpu(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  
  t0 = MPI_Wtime();
  for(int i=0; i<_NUM_ITERATIONS_GPU; ++i)
    transpose_naive_gpu(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  t = MPI_Wtime() - t0;
  
  // Transfer data from device

  pm->dev_pull(d_b, b, _NUM_ROWS * _NUM_COLS * sizeof(real_t));

  print_summary(t, _NUM_ROWS, _NUM_COLS, _NUM_ITERATIONS_GPU, "_transpose_naive_gpu");
  
  check_result(r, b, _NUM_ROWS*_NUM_COLS, "transpose_naive_gpu");

  //  print_matrix(b, _NUM_COLS, _NUM_ROWS, "Transposed b from naive_gpu");
  
  // ----------------------------------------------------------------
  
  // Execute kernel

  transpose_gpu_v1(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  
  t0 = MPI_Wtime();
  for(int i=0; i<_NUM_ITERATIONS_GPU; ++i)
    transpose_gpu_v1(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  t = MPI_Wtime() - t0;
  
  // Transfer data from device

  pm->dev_pull(d_b, b, _NUM_ROWS * _NUM_COLS * sizeof(real_t));

  print_summary(t, _NUM_ROWS, _NUM_COLS, _NUM_ITERATIONS_GPU, "_transpose_gpu");
  
  check_result(r, b, _NUM_ROWS*_NUM_COLS, "transpose_gpu");

  //  print_matrix(b, _NUM_COLS, _NUM_ROWS, "Transposed b from transpose_gpu");
  
  // ----------------------------------------------------------------
  
  // Execute kernel

  transpose_gpu_v2(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  
  t0 = MPI_Wtime();
  for(int i=0; i<_NUM_ITERATIONS_GPU; ++i)
    transpose_gpu_v2(d_b, d_a, _NUM_ROWS, _NUM_COLS);
  pm->dev_barrier();
  t = MPI_Wtime() - t0;
  
  // Transfer data from device

  pm->dev_pull(d_b, b, _NUM_ROWS * _NUM_COLS * sizeof(real_t));

  print_summary(t, _NUM_ROWS, _NUM_COLS, _NUM_ITERATIONS_GPU, "_transpose_gpu_2");
  
  check_result(r, b, _NUM_ROWS*_NUM_COLS, "transpose_gpu_2");

  //  print_matrix(b, _NUM_COLS, _NUM_ROWS, "Transposed b from transpose_gpu_2");
  
  // ----------------------------------------------------------------
  
  // Clean up

  pm->dev_free(d_a);
  pm->dev_free(d_b);
  
  delete pm;
    
  free(a);
  free(b);
  free(r);

  MPI_Finalize();
}
