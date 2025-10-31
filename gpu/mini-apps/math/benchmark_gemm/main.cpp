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

extern "C" {
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
              const int * k, const double * alpha, const double * a, const int * lda,
              const double * b, const int * ldb, const double * beta, double * c,
              const int * ldc);
}

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
 
#if 1
  dgemm_((const char*) "N", (const char*) "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc); 
#else 
  for(int i=0; i<m; ++i)
    for(int j=0; j<n; ++j) {
      double val = 0.0;
      for(int l=0; l<k; ++l) val += a[i*lda+l] * b[l*ldb+j];
      c[i*ldc+j] = alpha * val + beta * c[i*ldc+j];
    }
#endif
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

  input_t inp;

  parse_command_line(argc, argv, inp);
  
  real_t * a = (real_t*) malloc(inp.m * inp.k * inp.num_batches * sizeof(real_t));
  real_t * b = (real_t*) malloc(inp.k * inp.n * inp.num_batches * sizeof(real_t));
  
  real_t * c = (real_t*) malloc(inp.m * inp.n * inp.num_batches * sizeof(real_t));
  real_t * r = (real_t*) malloc(inp.m * inp.n * sizeof(real_t));

  std::srand(time(nullptr));
  
  // Initialize host

  for(int ib=0; ib<inp.num_batches; ++ib) {

    int offset = ib * inp.m * inp.k;
    for(int i=0; i<inp.m; ++i) {
      for(int j=0; j<inp.k; ++j) {
	a[i * inp.k + j] = std::rand() / (float(RAND_MAX) + 1.0) - 0.5;
      }
    }

    offset = ib * inp.k * inp.n;
    for(int i=0; i<inp.k; ++i) {
      for(int j=0; j<inp.n; ++j) {
	b[i * inp.n + j] = std::rand() / (float(RAND_MAX) + 1.0) - 0.5;
      }
    }
    
  }

  // ----------------------------------------------------------------
  // Naive CPU reference: Matrix Multiply
  // ----------------------------------------------------------------

  double t;

  printf("\nMatrix Multiplication :: C(%i x %i) = A(%i x %i)^%s . B(%i x %i)^%s\n",
	 inp.m, inp.n, inp.m, inp.k, inp.transa, inp.k, inp.n, inp.transb);

  if(inp.check_result) {
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      
      const int m = inp.m;  // # rows of first matrix A
      const int n = inp.n;  // # cols of second matrix B
      const int k = inp.k;  // # cols of first matrix A

      const int lda = inp.lda;
      const int ldb = inp.ldb;
      const int ldc = inp.ldc;
      
#if 1
  dgemm_((const char*) "N", (const char*) "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, r, &ldc);
#else  
    for(int i=0; i<inp.m; ++i)
      for(int j=0; j<inp.n; ++j) 	{
	double val = 0.0;
	for(int k=0; k<inp.k; ++k) val += a[i * inp.k + k] * b[k * inp.n + j];
	r[i * inp.n + j] = val;
      }
#endif

      gemm_NN0_naive_cpu(&m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      
      double t0 = MPI_Wtime();
      for(int i=0; i<_NUM_ITERATIONS_CPU; ++i)
	gemm_NN0_naive_cpu(&m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      t = MPI_Wtime() - t0;
    }
    
    print_summary(0, t, inp.m, inp.k, inp.n, _NUM_ITERATIONS_CPU, 1, "gemm_NN0_naive_cpu");
    
    check_result(r, c, inp.m*inp.n, "naive_cpu");
    
    //  print_matrix(a, inp.m, inp.k, "Original a");
    
    //  print_matrix(b, inp.k, inp.n, "Original b");
    
    //  print_matrix(r, inp.m, inp.n, "Reference r");
    
    //  print_matrix(c, inp.m, inp.n, "Output c");
    
  } // if(check_result)

  // overwrite c for next time

  for(int i=0; i<inp.m*inp.n*inp.num_batches; ++i) c[i] = -1.0;
  
  // ----------------------------------------------------------------
  // Optimized math library: Matrix Multiply
  // ----------------------------------------------------------------

  // Create device buffers and transfer data to device

  real_t ** d_a = (real_t **) pm->dev_malloc_host(num_devices * sizeof(real_t*));
  real_t ** d_b = (real_t **) pm->dev_malloc_host(num_devices * sizeof(real_t*));
  real_t ** d_c = (real_t **) pm->dev_malloc_host(num_devices * sizeof(real_t*));

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    d_a[i] = (real_t *) pm->dev_malloc(inp.m * inp.k * inp.num_batches * sizeof(real_t), "d_a[i]", FLERR);
    d_b[i] = (real_t *) pm->dev_malloc(inp.k * inp.n * inp.num_batches * sizeof(real_t), "d_b[i]", FLERR);
    d_c[i] = (real_t *) pm->dev_malloc(inp.m * inp.n * inp.num_batches * sizeof(real_t), "d_c[i]", FLERR);

    pm->dev_push(d_a[i], a, inp.m * inp.k * inp.num_batches * sizeof(real_t));
    pm->dev_push(d_b[i], b, inp.k * inp.n * inp.num_batches * sizeof(real_t));
  }
  
  const double alpha = 1.0;
  const double beta = 0.0;

  int m, n, k;
  int lda, ldb, ldc;
  int strideA, strideB, strideC;

  if(inp.fortran) {
  
    m = inp.n;  // # rows of first matrix B^T
    n = inp.m;  // # cols of second matrix A^T
    k = inp.k;  // # cols of first matrix B^T
    
    ldb = inp.ldb; // lead dimension of first matrix B^T
    lda = inp.lda; // lead dimension of second matrix A^T
    ldc = inp.ldc; // lead dimension of result matrix C
    
    strideA = inp.k * inp.n; // stride matrix B
    strideB = inp.m * inp.k; // stride matrix A
    strideC = inp.m * inp.n; // stride matrix C

  } else {

    m = inp.m;
    n = inp.n;
    k = inp.k;

    lda = inp.lda;
    ldb = inp.ldb;
    ldc = inp.ldc;

    strideA = inp.m * inp.k;
    strideB = inp.k * inp.m;
    strideC = inp.m * inp.n;
    
  }
    
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    ml->set_handle();

    if(inp.do_batched) {

      if(inp.fortran) {
	
	ml->gemm_batch(inp.transa, inp.transb, &m, &n, &k,
		       &alpha, d_b[i], &ldb, &strideB,
		       d_a[i], &lda, &strideA,
		       &beta, d_c[i], &ldc, &strideC,
		       &(inp.num_batches));

      } else {

	ml->gemm_batch(inp.transa, inp.transb, &m, &n, &k,
		       &alpha, d_a[i], &lda, &strideA,
		       d_b[i], &ldb, &strideB,
		       &beta, d_c[i], &ldc, &strideC,
		       &(inp.num_batches));
	
      }
      
    } else {
      
      if(inp.fortran) {

	ml->gemm(inp.transa, inp.transb, &m, &n, &k, &alpha, d_b[i], &ldb, d_a[i], &lda, &beta, d_c[i], &ldc);

      } else {
	
	ml->gemm(inp.transa, inp.transb, &m, &n, &k, &alpha, d_a[i], &lda, d_b[i], &ldb, &beta, d_c[i], &ldc);

      }

    }
  
    pm->dev_barrier();
  }

  double total_flops = 0.0;
  double total_flops2 = 0.0;

  double total_time = 0.0;
  double total_time2 = 0.0;
  
  for(int ir=0; ir<inp.num_repeat; ++ir) {
  
    double t0 = MPI_Wtime();
    for(int i=0; i<inp.num_iter; ++i) {
      
      for(int j=0; j<num_devices; ++j){
	pm->dev_set_device(j);
	ml->set_handle();

	if(inp.do_batched) {

	  if(inp.fortran) {
	    
	    ml->gemm_batch(inp.transa, inp.transb, &m, &n, &k,
			   &alpha, d_b[j], &ldb, &strideB,
			   d_a[j], &lda, &strideA,
			   &beta, d_c[j], &ldc, &strideC,
			   &(inp.num_batches));

	  } else {

	    ml->gemm_batch(inp.transa, inp.transb, &m, &n, &k,
			   &alpha, d_a[j], &lda, &strideA,
			   d_b[j], &ldb, &strideB,
			   &beta, d_c[j], &ldc, &strideC,
			   &(inp.num_batches));
	  
	  }

	} else {

	  if(inp.fortran) {
	    
	    ml->gemm(inp.transa, inp.transb, &m, &n, &k, &alpha, d_b[j], &ldb, d_a[j], &lda, &beta, d_c[j], &ldc);

	  } else {

	    ml->gemm(inp.transa, inp.transb, &m, &n, &k, &alpha, d_a[j], &lda, d_b[j], &ldb, &beta, d_c[j], &ldc);
	  
	  }

	}
      }
      
    }
    
    for(int i=0; i<num_devices; ++i) {
      pm->dev_set_device(i);
      pm->dev_barrier();
    }
    
    t = MPI_Wtime() - t0;

    total_time += t;
    total_time2 += t * t;
    
    for(int i=0; i<num_devices; ++i) {
      pm->dev_set_device(i);
      
      pm->dev_pull(d_c[i], c, inp.m * inp.n * inp.num_batches * sizeof(real_t));
      
      double flops = print_summary(i, t, inp.m, inp.k, inp.n, inp.num_iter, inp.num_batches, "MATHLIB gemm"); // GFlops

      total_flops += flops;
      total_flops2 += flops * flops;
      
      if(inp.check_result) check_result(r, c, inp.m*inp.n, "gemm_gpu");
      
      //  print_matrix(r, inp.m, inp.n, "Reference r");
      
      //  print_matrix(c, inp.m, inp.n, "Output c");
    }

  }

  double avg_flops = total_flops / double(inp.num_repeat) / double(num_devices);
  double avg_flops2 = total_flops2 / double(inp.num_repeat) / double(num_devices);
  double std_flops = sqrt(avg_flops2 - avg_flops*avg_flops + 1e-18);
  
  double avg_time = total_time / double(inp.num_repeat);
  double avg_time2 = total_time2 / double(inp.num_repeat);
  double std_time = sqrt(avg_time2 - avg_time*avg_time + 1e-18);
  
  printf("\n[MATHLIB gemm] %f +/- %f [ms]   %f +/- %f [TFlops]\n", avg_time*1000.0, std_time*1000.0, avg_flops/1000.0, std_flops/1000.0);
    
  // ----------------------------------------------------------------
  
  // Clean up

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    ml->destroy_handle();
    pm->dev_stream_destroy();

    pm->dev_free(d_a[i]);
    pm->dev_free(d_b[i]);
    pm->dev_free(d_c[i]);
  }
  
  delete ml;
  
  pm->dev_free_host(d_a);
  pm->dev_free_host(d_b);
  pm->dev_free_host(d_c);
  
  delete pm;
    
  free(a);
  free(b);
  free(c);
  free(r);

  MPI_Finalize();
}
