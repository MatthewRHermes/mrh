#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>
#include "mpi.h"

#include <omp.h>

#include "pm.h"

#define _N 1024
#define _LOCAL_SIZE 64

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

using namespace PM_NS;

// ----------------------------------------------------------------

void _vecadd(real_t * a, real_t * b, real_t * c, int N)
{

#pragma omp target teams distribute parallel for is_device_ptr(a, b, c) 
  for(int i=0; i<N; ++i) {
    c[i] = a[i] + b[i];
  }
  
}

// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  int me,nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  const int N = _N;

  class PM * pm = new PM();

  real_t * a = (real_t*) malloc(N*sizeof(real_t));
  real_t * b = (real_t*) malloc(N*sizeof(real_t));
  real_t * c = (real_t*) malloc(N*sizeof(real_t));

  // Initialize host
  for(int i=0; i<N; ++i) {
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
    c[i] = -1.0;
  }

  int num_devices = pm->dev_num_devices();

  if(me == 0) {
    printf("# of devices= %i\n",num_devices);
    pm->dev_properties(num_devices);
  }

  // Device ID

  int device_id = me % num_devices;
  for(int i=0; i<nranks; ++i) {
    if(i == me) {
      printf("Rank %i running on GPU %i!\n",me,device_id);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
#ifdef _SINGLE_PRECISION
  if(me == 0) printf("Using single-precision\n\n");
#else
  if(me == 0) printf("Using double-precision\n\n");
#endif

  // Create device buffers and transfer data to device

  real_t * d_a = (real_t *) pm->dev_malloc(N*sizeof(real_t));
  real_t * d_b = (real_t *) pm->dev_malloc(N*sizeof(real_t));
  real_t * d_c = (real_t *) pm->dev_malloc(N*sizeof(real_t));

  pm->dev_push(d_a, a, N);
  pm->dev_push(d_b, b, N);
  pm->dev_push(d_c, c, N);

  // Execute kernel

  _vecadd(d_a, d_b, d_c, N);

  // Transfer data from device

  pm->dev_pull(d_c, c, N);

  //Check result on host

  double diff = 0;
  for(int i=0; i<N; ++i) diff += (double) c[i];
  diff = diff/(double) N - 1.0;

  double diffsq = diff * diff;

  int sum;
  MPI_Reduce(&diffsq, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(me == 0) {
    if(sum < 1e-6) printf("\nResult is CORRECT!! :)\n");
    else printf("\nResult is WRONG!! :(\n");
  }

  // Clean up

  free(a);
  free(b);
  free(c);

  pm->dev_free(d_a);
  pm->dev_free(d_b);
  pm->dev_free(d_c);

  delete pm;

  MPI_Finalize();
}
