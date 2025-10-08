// Check achievable bandwidths for different GPU-related transfers
// Useful for confirming expectations and developing application performance projections
// -- Host-to-Device (H2D)
// -- Device-to-Device (D2D)
// -- D2D w/ peer-access enabled

// $ make ARCH=polaris update
// $ make ARCH=polaris
// $ mpiexec -n 4 ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include <mpi.h>
#include <omp.h>

#include "pm/pm.h"

#define _N_EXP 20 // 1 MB

#define _TOL 1e-8
#define _NUM_ITERATIONS 100

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

using namespace PM_NS;

// ----------------------------------------------------------------

void print_summary(double t, int n, int num_iter, const char * name)
{
  double size = n * sizeof(real_t) / 1024.0 / 1024.0; // MB

  double bandwidth = size * num_iter / t / 1024.0; // GB/s

  printf("P2P[%s] : n= %i num_iter= %i  size= %f  [MB]  time= %f  [ms]  bandwidth= %f  [GB/s]\n",
	 name, n, num_iter, size, t*1000.0, bandwidth);
}

// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  int me, nranks;
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

  const int device_id = me % num_devices;

  for(int i=0; i<nranks; ++i) {
    if(i == 0 && me == 0) printf("# of MPI ranks= %i  # of GPU devices= %i\n",nranks, num_devices);
    if(me == i) printf(" -- me= %i  device_id= %i\n",me,device_id);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  fflush(stdout);
  
  if(me == 0) pm->dev_properties(num_devices);

  pm->dev_set_device(device_id);
  
  const size_t _N = 1 << _N_EXP;
  
  const size_t size = _N * sizeof(real_t);
  
  real_t * a = (real_t *) pm->dev_malloc_host(size);

  // Initialize host
  
  for(int i=0; i<_N; ++i) a[i] = 2.0;

  // Initialize device

  real_t * d_a = (real_t *) pm->dev_malloc(size);
  real_t * d_b = (real_t *) pm->dev_malloc(size);
  
  // ----------------------------------------------------------------

  // Host-device transfers
  
  if(me == 0) printf("\nHost-to-device (H2D)\n");
  
  for(int i=0; i<nranks; ++i) {

    if(me == i) {
      double t0 = MPI_Wtime();
      
      for(int j=0; j<_NUM_ITERATIONS; ++j) pm->dev_push(d_a, a, size);
      
      double t = MPI_Wtime() - t0;
      
      char name[50];
      sprintf(name, "me= %i ; gpu= %i  H2D copy",me,device_id);
      print_summary(t, _N, _NUM_ITERATIONS, "H2D");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

  fflush(stdout);
  
  // ----------------------------------------------------------------

  // Device-device transfers
  
  if(me == 0) printf("\nDevice-to-device (D2D)\n");
  
  MPI_Status stat;

  for(int i=0; i<nranks; ++i) {

    for(int j=0; j<nranks; ++j) {

      if(i == j && i == me) {
	double t0 = MPI_Wtime();

	for(int k=0; k<_NUM_ITERATIONS; ++k) pm->dev_copy(d_b, d_a, size);

	double t = MPI_Wtime() - t0;

	char name[50];
	sprintf(name, "me= %i ; gpu= %i  D2D copy",me,device_id);
	
	print_summary(t, _N, _NUM_ITERATIONS, name);
      } else {
	double t0 = MPI_Wtime();

	for(int k=0; k<_NUM_ITERATIONS; ++k) {
	  if(i == me) MPI_Send(d_a, _N, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
	  if(j == me) MPI_Recv(d_b, _N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
	}

	double t = MPI_Wtime() - t0;
	
	char name[50];
	sprintf(name, "me= %i ; gpus= %i - %i  D2D transfer",me,i,j);
	
	if(i == me) print_summary(t, _N, _NUM_ITERATIONS, name);
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }
    printf("\n");
  }

  // Clean up

  pm->dev_free(d_a);
  pm->dev_free(d_b);
 
  pm->dev_free_host(a);
  
  delete pm;

  MPI_Finalize();
}
