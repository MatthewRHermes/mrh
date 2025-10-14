// A mini app to do reduce several vectors to a single vector (e.g. batched daxpy-like)
// This is largely used for benchmarking.

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

extern void my_gpu_vecadd(const real_t *, real_t *, int);
extern void my_gpu_vecadd_batch(const real_t *, real_t *, int, int);

// reduce_buf3_to_rdmInside mrh::mole.py adding use_gpu flag
// reduce_buf3_to_rdm : size= 20736  num_batches= 132
// reduce_buf3_to_rdm : size= 20736  num_batches= 33
// reduce_buf3_to_rdm : size= 28561  num_batches= 11
// reduce_buf3_to_rdm : size= 28561  num_batches= 30
// reduce_buf3_to_rdm : size= 28561  num_batches= 32
// reduce_buf3_to_rdm : size= 28561  num_batches= 77

// -n [# elements in vector] -batch [# of vectors to accumulate]

// vector addition (daxpy)
// OMP_NUM_THREADS=1 ./a.out -replay gemm N N  3 4 2  2 4 4  1.0 0.0 -check_result -fortran-order

// ----------------------------------------------------------------

struct input_t {
  bool check_result = false;
  bool do_batched = false;
  bool fortran = false;
  int num_batches = 132;
  int num_iter = 100;
  int num_repeat = 1;
  int n = 20736;
};

// ----------------------------------------------------------------

void parse_command_line(int argc, char * argv[], input_t & inp)
{
  //  printf("argc= %i\n",argc);
  //  for(int i=0; i<argc; ++i) printf("i= %i  argv= %s\n",i,argv[i]);

  int indx = 1;
  while(indx < argc) {

    if(strcmp(argv[indx], "-check_result") == 0) inp.check_result = true;
    else if(strcmp(argv[indx], "-n") == 0) {
      inp.n = atoi(argv[++indx]);
    }
    else if(strcmp(argv[indx], "-num_iter") == 0) inp.num_iter = atoi(argv[++indx]);
    else if(strcmp(argv[indx], "-batched") == 0) inp.do_batched = true;
    else if(strcmp(argv[indx], "-num_batches") == 0) {
      inp.num_batches = atoi(argv[++indx]);
      inp.do_batched = true;
    }
    else if(strcmp(argv[indx], "-num_repeat") == 0) inp.num_repeat = atoi(argv[++indx]);
    else if(strcmp(argv[indx], "-fortran-order") == 0) inp.fortran = true;

    indx++;
  }

  printf("\ninput_params\n");
  printf("------------\n");
  printf("check_result= %i\n",inp.check_result);
  printf("n= %i %i %i\n",inp.n);
  printf("num_iter= %i\n",inp.num_iter);
  printf("do_batched= %i\n",inp.do_batched);
  printf("num_batches= %i\n",inp.num_batches);
  printf("num_repeat= %i\n",inp.num_repeat);
}

// ----------------------------------------------------------------

void vecadd_naive_cpu(real_t * in, real_t * out, const int * n_, const int * num_batches_)
{
  int n = *n_;
  int num_batches = *num_batches_;

  for(int i=0; i<n; ++i) out[i] = 0.0;
  
  for(int i=0; i<num_batches; ++i) {
    real_t * array = in + i*n;
    for(int j=0; j<n; ++j) out[j] += array[j];
  }
}

// ----------------------------------------------------------------

void vecadd_naive_gpu(real_t * in, real_t * out, const int * n_, const int * num_batches_)
{
  int n = *n_;
  int num_batches = *num_batches_;

  for (int i=0; i<num_batches; ++i) my_gpu_vecadd(&(in[i*n]), out, n);
}

// ----------------------------------------------------------------

void vecadd_batch_gpu(real_t * in, real_t * out, const int * n_, const int * num_batches_)
{
  int n = *n_;
  int num_batches = *num_batches_;

  my_gpu_vecadd_batch(in, out, n, num_batches);
}

// ----------------------------------------------------------------

void vecadd_daxpy_gpu(MATHLIB * ml, real_t * in, real_t * out, const int * n_, const int * num_batches_)
{
  int n = *n_;
  int num_batches = *num_batches_;

  const real_t alpha = 1.0;
  const int inc = 1;
  
  for (int i=0; i<num_batches; ++i) ml->axpy(&n, &alpha, &(in[i*n]), &inc, out, &inc);
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

void print_vector(real_t * data, int num_rows, const char * name)
{
  printf("\nVector[%s] : %i x 1 \n",name, num_rows);
  for(int i=0; i<num_rows; ++i) printf(" %f", data[i]);
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

void test(int argc, char ** argv)
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

  input_t inp;

  parse_command_line(argc, argv, inp);

  const int size_A = inp.n * inp.num_batches * sizeof(real_t);
  const int size_B = inp.n * sizeof(real_t);
  const int size_C = inp.n * sizeof(real_t);
  
  real_t * a = (real_t*) malloc(size_A);
  real_t * b = (real_t*) malloc(size_B);
  real_t * r = (real_t*) malloc(size_C);
  
  // Initialize host
  
  for(int i=0; i<inp.num_batches; ++i) {
    for(int j=0; j<inp.n; ++j) a[i*inp.n + j] = 1.0;
  }

  for(int i=0; i<inp.n; ++i) r[i] = (real_t) inp.num_batches;
  
  // ----------------------------------------------------------------
  // Naive CPU reference: vector addition
  // ----------------------------------------------------------------

  double total_time = 0.0;
  double total_time2 = 0.0;
  
  double t;

  printf("\nVector Addition :: n= %i  num_batches= %i\n", inp.n, inp.num_batches);

  for(int ir=0; ir<inp.num_repeat; ++ir) {
    double t0 = MPI_Wtime();
    for(int i=0; i<inp.num_iter; ++i) {
      
      vecadd_naive_cpu(a, b, &(inp.n), &(inp.num_batches));

    }

    t = MPI_Wtime() - t0;

    total_time += t;
    total_time2 += t * t;

  }
  
  check_result(r, b, inp.n, "naive_cpu");

  double avg_time = total_time / double(inp.num_repeat) / double(inp.num_iter);
  double avg_time2 = total_time2 / double(inp.num_repeat) / double(inp.num_iter);

  double std_time = sqrt(avg_time2 - avg_time*avg_time + 1e-18);

  printf("[CPU NAIVE] %f +/- %f [ms]\n\n", avg_time*1000.0, std_time*1000.0);
  
  //  print_vector(b, inp.n, "Original b");
    
  //  print_vector(r, inp.n, "Reference r");
  
  // Create device buffers and transfer data to device

  real_t * d_a = (real_t *) pm->dev_malloc(size_A, "d_a", FLERR);
  real_t * d_b = (real_t *) pm->dev_malloc(size_B, "d_b", FLERR);

  pm->dev_push(d_a, a, size_A);
  
  // ----------------------------------------------------------------
  // GPU : vector addition w/ naive kernel + serialized
  // ----------------------------------------------------------------

  total_time = 0.0;
  total_time2 = 0.0;
  
  for(int ir=0; ir<inp.num_repeat; ++ir) {

    for(int i=0; i<inp.n; ++i) b[i] = 0.0;
    pm->dev_push(d_b, b, size_B);
    
    double t0 = MPI_Wtime();
    for(int i=0; i<inp.num_iter; ++i) {
      
      vecadd_naive_gpu(d_a, d_b, &(inp.n), &(inp.num_batches));

    }

    pm->dev_barrier();
    t = MPI_Wtime() - t0;

    total_time += t;
    total_time2 += t * t;

  }
  
  pm->dev_barrier();

  pm->dev_pull(d_b, b, size_B);

  for(int i=0; i<inp.n; ++i) b[i] /= (double) inp.num_iter;
  
  check_result(r, b, inp.n, "naive_gpu");
  
  avg_time = total_time / double(inp.num_repeat) / double(inp.num_iter);
  avg_time2 = total_time2 / double(inp.num_repeat) / double(inp.num_iter);

  std_time = sqrt(avg_time2 - avg_time*avg_time + 1e-18);

  printf("[GPU NAIVE] %f +/- %f [ms]\n\n", avg_time*1000.0, std_time*1000.0);
  
  // ----------------------------------------------------------------
  // GPU : vector addition w/ daxpy + serialized
  // ----------------------------------------------------------------

  total_time = 0.0;
  total_time2 = 0.0;
  
  for(int ir=0; ir<inp.num_repeat; ++ir) {

    for(int i=0; i<inp.n; ++i) b[i] = 0.0;
    pm->dev_push(d_b, b, size_B);
    
    double t0 = MPI_Wtime();
    for(int i=0; i<inp.num_iter; ++i) {
      
      vecadd_daxpy_gpu(ml, d_a, d_b, &(inp.n), &(inp.num_batches));

    }

    pm->dev_barrier();
    t = MPI_Wtime() - t0;

    total_time += t;
    total_time2 += t * t;

  }
  
  pm->dev_barrier();

  pm->dev_pull(d_b, b, size_B);

  for(int i=0; i<inp.n; ++i) b[i] /= (double) inp.num_iter;
  
  check_result(r, b, inp.n, "daxpy_gpu");
  
  avg_time = total_time / double(inp.num_repeat) / double(inp.num_iter);
  avg_time2 = total_time2 / double(inp.num_repeat) / double(inp.num_iter);

  std_time = sqrt(avg_time2 - avg_time*avg_time + 1e-18);

  printf("[GPU DAXPY] %f +/- %f [ms]\n\n", avg_time*1000.0, std_time*1000.0);
  
  // ----------------------------------------------------------------
  // GPU : vector addition w/ naive kernel in single offload
  // ----------------------------------------------------------------

  total_time = 0.0;
  total_time2 = 0.0;
  
  for(int ir=0; ir<inp.num_repeat; ++ir) {

    for(int i=0; i<inp.n; ++i) b[i] = 0.0;
    pm->dev_push(d_b, b, size_B);
    
    double t0 = MPI_Wtime();
    for(int i=0; i<inp.num_iter; ++i) {
      
      vecadd_batch_gpu(d_a, d_b, &(inp.n), &(inp.num_batches));

    }

    pm->dev_barrier();
    t = MPI_Wtime() - t0;

    total_time += t;
    total_time2 += t * t;

  }
  
  pm->dev_barrier();

  pm->dev_pull(d_b, b, size_B);

  for(int i=0; i<inp.n; ++i) b[i] /= (double) inp.num_iter;
  
  check_result(r, b, inp.n, "daxpy_gpu");
  
  avg_time = total_time / double(inp.num_repeat) / double(inp.num_iter);
  avg_time2 = total_time2 / double(inp.num_repeat) / double(inp.num_iter);

  std_time = sqrt(avg_time2 - avg_time*avg_time + 1e-18);

  printf("[GPU DAXPY] %f +/- %f [ms]\n\n", avg_time*1000.0, std_time*1000.0);
  
  // Clean up
    
  ml->destroy_handle();
  pm->dev_stream_destroy();

  pm->dev_free(d_a);
  pm->dev_free(d_b);
  
  delete ml;
  
  delete pm;
    
  free(a);
  free(b);
  free(r);
}

// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  test(argc, argv);

  MPI_Finalize();
}
