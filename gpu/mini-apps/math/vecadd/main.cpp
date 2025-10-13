// A mini app to do matrix multiplication on cpu and gpu
// This is largely used now to benchmark transforms recorded from production run

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <cassert>

#include <mpi.h>
#include <omp.h>

#include "pm/pm.h"
#include "mathlib/mathlib.h"

#define _TOL 1e-6
#define _NUM_ITERATIONS_CPU 1

using namespace PM_NS;
using namespace MATHLIB_NS;

// A is (m, k) matrix
// B is (k, n) matrix
// C is (m, n) matrix

// -replay to rerun workloads sampled from gpu4mrh run exactly as they were executed
// -replay [gemm|gemm_batch] [transa] [transb] [m] [n] [k] [lda] [ldb] [ldc] [alpha] [beta] [batch_size]
// -replay gemm_batch T T 92 92 92 1 0 240
// -replay gemm N N 92 92 22080 1 0

// -fortran-order : use this with by-hand testing
// Column-ordering transposes everything
// To compute A.B, then call Fortran APIs with B.A

// Matrix Multiplication :: C(3 x 4) = A(3 x 2)^N . B(2 x 4)^N
// OMP_NUM_THREADS=1 ./a.out -replay gemm N N  3 4 2  2 4 4  1.0 0.0 -check_result -fortran-order

// ----------------------------------------------------------------

struct input_t {
  bool check_result = false;
  bool do_batched = false;
  bool fortran = false;
  int num_batches = 1;
  int num_iter = 100;
  int num_repeat = 1;
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int lda = 1024;
  int ldb = 1024;
  int ldc = 1024;
  double alpha = 1.0;
  double beta = 0.0;
  char * transa = (char *) "N";
  char * transb = (char *) "N";
};

// ----------------------------------------------------------------

void parse_command_line(int argc, char * argv[], input_t & inp)
{
  //  printf("argc= %i\n",argc);
  //  for(int i=0; i<argc; ++i) printf("i= %i  argv= %s\n",i,argv[i]);

  int indx = 1;
  while(indx < argc) {

    if(strcmp(argv[indx], "-check_result") == 0) inp.check_result = true;
    else if(strcmp(argv[indx], "-mnk") == 0) {
      inp.m = atoi(argv[++indx]);
      inp.n = atoi(argv[++indx]);
      inp.k = atoi(argv[++indx]);
    }
    else if(strcmp(argv[indx], "-trans") == 0) {
      inp.transa = argv[++indx];
      inp.transb = argv[++indx];
      
      if(strcmp(inp.transa, "N") == 0) inp.lda = inp.k;
      else inp.lda = inp.m;

      if(strcmp(inp.transb, "N") == 0) {
	inp.ldb = inp.n;
	inp.ldc = inp.n;
      } else {
	inp.ldb = inp.k;
	inp.ldc = inp.k;
      }
    }
    else if(strcmp(argv[indx],"-replay") == 0) {
      if(strcmp(argv[++indx], "gemm_batch") == 0) inp.do_batched = true;
      inp.transa = argv[++indx];
      inp.transb = argv[++indx];
      inp.m = atoi(argv[++indx]);
      inp.n = atoi(argv[++indx]);
      inp.k = atoi(argv[++indx]);
      inp.lda = atoi(argv[++indx]);
      inp.ldb = atoi(argv[++indx]);
      inp.ldc = atoi(argv[++indx]);
      inp.alpha = atof(argv[++indx]);
      inp.beta = atof(argv[++indx]);
      if(inp.do_batched) inp.num_batches = atoi(argv[++indx]);
    }
    else if(strcmp(argv[indx], "-num_iter") == 0) inp.num_iter = atoi(argv[++indx]);
    else if(strcmp(argv[indx], "-batched") == 0) inp.do_batched = true;
    else if(strcmp(argv[indx], "-num_batches") == 0) inp.num_batches = atoi(argv[++indx]);
    else if(strcmp(argv[indx], "-num_repeat") == 0) inp.num_repeat = atoi(argv[++indx]);
    else if(strcmp(argv[indx], "-fortran-order") == 0) inp.fortran = true;

    indx++;
  }

  printf("\ninput_params\n");
  printf("------------\n");
  printf("check_result= %i\n",inp.check_result);
  printf("trans= %s %s\n",inp.transa, inp.transb);
  printf("mnk= %i %i %i\n",inp.m, inp.n, inp.k);
  printf("ld= %i %i %i\n",inp.lda, inp.ldb, inp.ldc);
  printf("num_iter= %i\n",inp.num_iter);
  printf("do_batched= %i\n",inp.do_batched);
  printf("num_batches= %i\n",inp.num_batches);
  printf("num_repeat= %i\n",inp.num_repeat);
  printf("fortran= %i\n",inp.fortran);
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
  double max_diff = 0.0;
  for(int i=0; i<n; ++i) {
    real_t diff = (ref[i] - test[i]) * (ref[i] - test[i]);
    //    printf(" -- i= %i  ref= %f  test= %f  diff= %f\n",i,ref[i],test[i],diff);
    //    if(diff > _TOL) err++;
    if(diff > max_diff) max_diff = diff;
  }

  if(max_diff > _TOL) err = 1;
  
  if(err == 0) printf("Results from %s are correct!! :)\n", name);
  else printf("Results from %s are incorrect!! :(  max_diff= %0.4e\n", name, max_diff);
  
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

double print_summary(int id, double t, int num_rows_a, int num_cols_a, int num_cols_b, int num_iter, int num_batch, const char * name)
{
  double flop = num_batch * num_rows_a * num_cols_b * (2.0 * num_cols_a - 1.0) / 1024.0 / 1024.0 / 1024.0; // TFlop

  double flops = flop * num_iter / t; // GFlop/s

  printf("\nMatrix[%s : %i] : mnk= %i %i %i  num_batch= %i  num_iter= %i  time= %f  [ms]  flops= %f  [GFlops/s]\n",
	 name, id, num_rows_a, num_cols_b, num_cols_a, num_batch, num_iter, t*1000.0, flops);

  fflush(stdout);
  
  return flops;
}

// ----------------------------------------------------------------

void test_NN(int argc, char ** argv)
{
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
  
  int num_devices = pm->dev_num_devices();

  class MATHLIB * ml = new MATHLIB(pm);
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    int hid = ml->create_handle();
  }

  if(me == 0) {
    printf("\n# of devices= %i\n",num_devices);
    pm->dev_properties(num_devices);
  }
  
  // Device ID

  int device_id = me % num_devices;

  pm->dev_set_device(device_id);
  
  ml->set_handle();
  
  for(int i=0; i<nranks; ++i) {
    if(i == me) {
      printf("Rank %i running on GPU %i!\n",me,device_id);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //  input_t inp;

  //  parse_command_line(argc, argv, inp);

  const int num_rows_A = 4;
  const int num_cols_A = 2;

  const int num_rows_B = num_cols_A;
  const int num_cols_B = 3;

  const int num_rows_C = num_rows_A;
  const int num_cols_C = num_cols_B;

  const int size_A = num_rows_A * num_cols_A * sizeof(real_t);
  const int size_B = num_rows_B * num_cols_B * sizeof(real_t);
  const int size_C = num_rows_C * num_cols_C * sizeof(real_t);
  
  real_t * a = (real_t*) malloc(size_A);
  real_t * b = (real_t*) malloc(size_B);
  
  real_t * c = (real_t*) malloc(size_C);
  real_t * r = (real_t*) malloc(size_C);

  std::srand(time(nullptr));
  
  // Initialize host
  
  for(int i=0; i<num_rows_A; ++i) {
    for(int j=0; j<num_cols_A; ++j) {
      a[i * num_cols_A + j] = std::rand() / (float(RAND_MAX) + 1.0) - 0.5;
    }
  }

  for(int i=0; i<num_rows_B; ++i) {
    for(int j=0; j<num_cols_B; ++j) {
      b[i * num_cols_B + j] = std::rand() / (float(RAND_MAX) + 1.0) - 0.5;
    }
  }
  
  // ----------------------------------------------------------------
  // Naive CPU reference: Matrix Multiply
  // ----------------------------------------------------------------

  double t;

  printf("\nMatrix Multiplication :: C(%i x %i) = A(%i x %i)^N . B(%i x %i)^N\n",
	 num_rows_C, num_cols_C, num_rows_A, num_cols_A, num_rows_B, num_cols_B);
  
  printf("\n                      :: C(%i x %i) = A(%i x %i)   . B(%i x %i)\n",
	 num_rows_C, num_cols_C, num_rows_A, num_cols_A, num_rows_B, num_cols_B);
  
  for(int i=0; i<num_rows_C; ++i)
    for(int j=0; j<num_cols_C; ++j) {
      double val = 0.0;
      for(int l=0; l<num_rows_A; ++l) val += a[i * num_cols_A + l] * b[l * num_cols_B + j];
      r[i * num_cols_C + j] = val;
    }
    
  check_result(r, c, num_rows_C*num_cols_C, "naive_cpu");
    
  print_matrix(a, num_rows_A, num_rows_B, "Original a");
    
  print_matrix(b, num_rows_B, num_cols_B, "Original b");
    
  print_matrix(r, num_rows_C, num_cols_C, "Reference r");
  
  // ----------------------------------------------------------------
  // Optimized math library: Matrix Multiply
  // ----------------------------------------------------------------

  // Create device buffers and transfer data to device

  real_t * d_a = (real_t *) pm->dev_malloc(size_A);
  real_t * d_b = (real_t *) pm->dev_malloc(size_B);
  real_t * d_c = (real_t *) pm->dev_malloc(size_C);

  pm->dev_push(d_a, a, size_A);
  pm->dev_push(d_b, b, size_B);

  double alpha = 1.0;
  double beta = 0.0;

  int m = num_rows_B;  // # rows of first matrix B^T
  int n = num_rows_A;  // # cols of second matrix A^T
  int k = num_rows_B;  // # cols of first matrix B^T
  
  int lda = m; // lead dimension of first matrix B^T
  int ldb = k; // lead dimension of second matrix A^T
  int ldc = m; // lead dimension of result matrix C
  
  ml->gemm((char *) "N", (char *) "N", &m, &n, &k, &alpha, d_b, &ldb, d_a, &lda, &beta, d_c, &ldc);
  
  pm->dev_barrier();

  pm->dev_pull(d_c, c, size_C);

  print_matrix(c, num_rows_C, num_cols_C, "Output c");
  
  // Clean up
    
  ml->destroy_handle();
  pm->dev_stream_destroy();

  pm->dev_free(d_a);
  pm->dev_free(d_b);
  pm->dev_free(d_c);
  
  delete ml;
  
  delete pm;
    
  free(a);
  free(b);
  free(c);
  free(r);
}

// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  test_NN(argc, argv);

  MPI_Finalize();
}
