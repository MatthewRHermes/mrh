#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include <mpi.h>
#include <omp.h>

#include "pm.h"
#include "mathlib.h"

#define _NUM_BATCHES 100

#define _NUM_ROWS_A 4
#define _NUM_COLS_A 3

#define _NUM_ROWS_B _NUM_COLS_A
#define _NUM_COLS_B 5

#define _TOL 1e-8
#define _NUM_ITERATIONS_CPU 10
#define _NUM_ITERATIONS_GPU 1000

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

using namespace PM_NS;
using namespace MATHLIB_NS;

// A is (m, k) matrix
// B is (k, n) matrix
// C is (m, n) matrix

// Column-ordering transposes everything
// To compute A.B, then to call API with B.A

extern "C" {
  void dsymm_(const char*, const char*, const int*, const int*,
	      const double*, const double*, const int*,
	      const double*, const int*,
	      const double*, double*, const int*);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
	      const int * k, const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb, const double * beta, double * c,
	      const int * ldc);
}

// ----------------------------------------------------------------

void gemm_NN0_naive_cpu(const int * m_, const int * n_, const int * k_, const real_t * alpha_,
			real_t * a, const int * lda_, real_t * b, const int * ldb_,
			const real_t * beta_, real_t * c, const int * ldc_)
{
  double alpha = *alpha_;
  double beta = *beta_;
  
  int m = *m_;
  int n = *n_;
  int k = *k_;

  int lda = *lda_;
  int ldb = *ldb_;
  int ldc = *ldc_;
  
  for(int i=0; i<m; ++i)
    for(int j=0; j<n; ++j) {
      double val = 0.0;
      for(int l=0; l<k; ++l) val += a[i*lda+l] * b[l*ldb+j];
      c[i*ldc+j] = alpha * val + beta * c[i*ldc+j];
    }
}

// ----------------------------------------------------------------

int check_result(real_t * ref, real_t * test, int n, const char * name)
{
  int err = 0;
  for(int i=0; i<n; ++i) {
    real_t diff = (ref[i] - test[i]) * (ref[i] - test[i]);
    //    printf(" -- i= %i  ref= %f  test= %f  diff= %f\n",i,ref[i],test[i],diff);
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

void print_summary(double t, int num_rows_a, int num_cols_a, int num_cols_b, int num_iter, const char * name)
{
  double flop = 2.0 * num_rows_a * num_cols_a * num_cols_b / 1024.0 / 1024.0 / 1024.0; // GFlop

  double flops = flop * num_iter / t; // GFlop/s

  printf("\nMatrix[%s] : mnk= %i %i %i  num_iter= %i  time= %f  [ms]  flops= %f  [GB/s]\n",
	 name, num_rows_a, num_cols_b, num_cols_a, num_iter, t*1000.0, flops);
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

  class MATHLIB * ml = new MATHLIB();
  
  int num_devices = pm->dev_num_devices();

  if(me == 0) {
    printf("\n# of devices= %i\n",num_devices);
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

  real_t * a = (real_t*) malloc(_NUM_ROWS_A * _NUM_COLS_A * sizeof(real_t));
  real_t * b = (real_t*) malloc(_NUM_ROWS_B * _NUM_COLS_B * sizeof(real_t));
  
  real_t * c = (real_t*) malloc(_NUM_ROWS_A * _NUM_COLS_B * sizeof(real_t));
  real_t * r = (real_t*) malloc(_NUM_ROWS_A * _NUM_COLS_B * sizeof(real_t));

  // Initialize host
  
  for(int i=0; i<_NUM_ROWS_A; ++i) {
    for(int j=0; j<_NUM_COLS_A; ++j) {
      a[i*_NUM_COLS_A + j] = (i * _NUM_COLS_A + j) * 0.1;
    }
  }
  
  for(int i=0; i<_NUM_ROWS_B; ++i) {
    for(int j=0; j<_NUM_COLS_B; ++j) {
      b[i*_NUM_COLS_B + j] = (i * _NUM_COLS_B + j) * 0.1;
    }
  }

  // ----------------------------------------------------------------
  // Naive CPU reference: Matrix Multiply
  // ----------------------------------------------------------------

  printf("\nMatrix Multiplication :: C(%i x %i) = A(%i x %i).B(%i x %i)\n",
	 _NUM_ROWS_A, _NUM_COLS_B, _NUM_ROWS_A, _NUM_COLS_A, _NUM_ROWS_B, _NUM_COLS_B);
  
  for(int i=0; i<_NUM_ROWS_A; ++i)
    for(int j=0; j<_NUM_COLS_B; ++j)
      {
	double val = 0.0;
	for(int k=0; k<_NUM_COLS_A; ++k) val += a[i*_NUM_COLS_A+k] * b[k*_NUM_COLS_B+j];
	r[i*_NUM_COLS_B+j] = val;
      }

  double t;
  
  {
    const double alpha = 1.0;
    const double beta = 0.0;

    const int m = _NUM_ROWS_A;  // # rows of first matrix A
    const int n = _NUM_COLS_B;  // # cols of second matrix B
    const int k = _NUM_COLS_A;  // # cols of first matrix A

    const int lda = _NUM_COLS_A; // lead dimension of first matrix A
    const int ldb = _NUM_COLS_B; // lead dimension of second matrix B
    const int ldc = _NUM_COLS_B; // lead dimension of result matrix C
  
    gemm_NN0_naive_cpu(&m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

    double t0 = MPI_Wtime();
    for(int i=0; i<_NUM_ITERATIONS_CPU; ++i)
      gemm_NN0_naive_cpu(&m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    t = MPI_Wtime() - t0;
  }

  print_summary(t, _NUM_ROWS_A, _NUM_COLS_A, _NUM_COLS_B, _NUM_ITERATIONS_CPU, "gemm_NN0_naive_cpu");

  check_result(r, c, _NUM_ROWS_A*_NUM_COLS_B, "naive_cpu");
  
  //  print_matrix(a, _NUM_ROWS_A, _NUM_COLS_A, "Original a");

  //  print_matrix(b, _NUM_ROWS_B, _NUM_COLS_B, "Original b");

  //  print_matrix(r, _NUM_ROWS_A, _NUM_COLS_B, "Reference r");
  
  //  print_matrix(c, _NUM_ROWS_A, _NUM_COLS_B, "Output c");

  // ----------------------------------------------------------------
  // Optimized math library: Matrix Multiply
  // ----------------------------------------------------------------

  // Create device buffers and transfer data to device
  
  real_t * d_a = (real_t *) pm->dev_malloc(_NUM_ROWS_A * _NUM_COLS_A * sizeof(real_t));
  real_t * d_b = (real_t *) pm->dev_malloc(_NUM_ROWS_B * _NUM_COLS_B * sizeof(real_t));
  real_t * d_c = (real_t *) pm->dev_malloc(_NUM_ROWS_A * _NUM_COLS_B * sizeof(real_t));

  pm->dev_push(d_a, a, _NUM_ROWS_A * _NUM_COLS_A * sizeof(real_t));
  pm->dev_push(d_b, b, _NUM_ROWS_B * _NUM_COLS_B * sizeof(real_t));

#if defined(_USE_CPU)
  
  {
    void * handle;
    
    const double alpha = 1.0;
    const double beta = 0.0;

    const int m = _NUM_COLS_B;  // # rows of first matrix B^T
    const int n = _NUM_ROWS_A;  // # cols of second matrix A^T
    const int k = _NUM_ROWS_B;  // # cols of first matrix B^T

    const int ldb = _NUM_COLS_B; // lead dimension of first matrix B^T
    const int lda = _NUM_COLS_A; // lead dimension of second matrix A^T
    const int ldc = _NUM_COLS_B; // lead dimension of result matrix C^T

    ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, d_b, &ldb, d_a, &lda, &beta, d_c, &ldc, handle);

    pm->dev_barrier();
    
    double t0 = MPI_Wtime();
    for(int i=0; i<_NUM_ITERATIONS_CPU; ++i)
      ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, d_b, &ldb, d_a, &lda, &beta, d_c, &ldc, handle);
    pm->dev_barrier();
    t = MPI_Wtime() - t0;
  }
#endif

#if defined(_USE_GPU)

#if defined(_GPU_CUDA)

  cudaStream_t stream;

  pm->dev_stream_create(stream);
  
  cublasHandle_t handle;
  
  cublasCreate(&handle);
  
  cublasSetStream(handle, stream);

  const double alpha = 1.0;
  const double beta = 0.0;
  
  const int m = _NUM_COLS_B;  // # rows of first matrix B^T
  const int n = _NUM_ROWS_A;  // # cols of second matrix A^T
  const int k = _NUM_ROWS_B;  // # cols of first matrix B^T
  
  const int ldb = _NUM_COLS_B; // lead dimension of first matrix B^T
  const int lda = _NUM_COLS_A; // lead dimension of second matrix A^T
  const int ldc = _NUM_COLS_B; // lead dimension of result matrix C^T

  ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, d_b, &ldb, d_a, &lda, &beta, d_c, &ldc, handle);
  
  pm->dev_barrier();
  
  double t0 = MPI_Wtime();
  for(int i=0; i<_NUM_ITERATIONS_CPU; ++i)
    ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, d_b, &ldb, d_a, &lda, &beta, d_c, &ldc, handle);
  pm->dev_barrier();
  t = MPI_Wtime() - t0;
#endif

#endif
  
  pm->dev_pull(d_c, c, _NUM_ROWS_A * _NUM_COLS_B * sizeof(real_t));
  
  print_summary(t, _NUM_ROWS_A, _NUM_COLS_A, _NUM_COLS_B, _NUM_ITERATIONS_CPU, "LAPACK gemm");

  check_result(r, c, _NUM_ROWS_A*_NUM_COLS_B, "lapack_dgemm_cpu");

  //  print_matrix(r, _NUM_ROWS_A, _NUM_COLS_B, "Reference r");
  
  //  print_matrix(c, _NUM_ROWS_A, _NUM_COLS_B, "Output c");
  
  // ----------------------------------------------------------------
  
  // Clean up

#if defined(_USE_GPU)
#if defined(_GPU_CUDA)
  cublasDestroy(handle);
  pm->dev_stream_destroy(stream);
#endif
#endif
  
  delete ml;
  
  pm->dev_free(d_a);
  pm->dev_free(d_b);
  pm->dev_free(d_c);
  
  delete pm;
    
  free(a);
  free(b);
  free(c);
  free(r);

  MPI_Finalize();
}
