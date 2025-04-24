// Check achievable bandwidths for different GPU-related transfers
// Useful for confirming expectations and developing application performance projections
// -- Host-to-Device (H2D)
// -- Device-to-Device (D2D)
// -- D2D w/ peer-access enabled

// $ make ARCH=polaris update
// $ make ARCH=polaris
// $ ./a.out
// $ nsys profile --stats=true ./a.out

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
#define _NUM_ITERATIONS 10

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

  if(me == 0) {
    printf("# of devices= %i\n",num_devices);
    pm->dev_properties(num_devices);
  }

  const size_t _N = 1 << _N_EXP;
  
  const size_t size = _N * sizeof(real_t);
  
  real_t * a = (real_t *) pm->dev_malloc_host(size);

  // Initialize host
  
  for(int i=0; i<_N; ++i) a[i] = 2.0;

  // Initialize device

  real_t ** d_a = (real_t **) pm->dev_malloc_host(num_devices * sizeof(real_t *));
  real_t ** d_b = (real_t **) pm->dev_malloc_host(num_devices * sizeof(real_t *));

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    d_a[i] = (real_t *) pm->dev_malloc(size);
    d_b[i] = (real_t *) pm->dev_malloc(size);
  }
  
  // ----------------------------------------------------------------

  // Host-device transfers

  printf("\nHost-to-device (H2D)\n");

  pm->dev_profile_start("Host-to-Device (H2D)");
  
  for(int i=0; i<num_devices; ++i) {
    
    pm->dev_set_device(i);
  
    double t0 = MPI_Wtime();
  
    for(int j=0; j<_NUM_ITERATIONS; ++j) pm->dev_push(d_a[i], a, size);
  
    double t = MPI_Wtime() - t0;
    
    print_summary(t, _N, _NUM_ITERATIONS, "H2D");
  }

  pm->dev_profile_stop();
  
  // ----------------------------------------------------------------

  // Device-device transfers
  
  printf("\nDevice-to-device (D2D)\n");

  pm->dev_profile_start("Device-to-device (D2D)");
  
  for(int i=0; i<num_devices; ++i) {

    pm->dev_set_device(i);
    
    for(int j=0; j<num_devices; ++j) {

      if(i == j) {
	double t0 = MPI_Wtime();

	for(int k=0; k<_NUM_ITERATIONS; ++k) pm->dev_copy(d_b[j], d_a[i], size);

	double t = MPI_Wtime() - t0;

	char name[50];
	sprintf(name, "%i - %i  D2D copy",i,i);
	
	print_summary(t, _N, _NUM_ITERATIONS, name);
      } else {
	double t0 = MPI_Wtime();

	for(int k=0; k<_NUM_ITERATIONS; ++k) pm->dev_copy(d_b[j], d_a[i], size);

	double t = MPI_Wtime() - t0;
	
	char name[50];
	sprintf(name, "%i - %i  D2D transfer",i,j);
	
	print_summary(t, _N, _NUM_ITERATIONS, name);
      }

      pm->dev_barrier();

    }
    printf("\n");
  }

  pm->dev_profile_stop();
  
  // ----------------------------------------------------------------

  // Enable peer-to-peer access

  int peer_error = pm->dev_check_peer(0, num_devices);
  if(!peer_error) pm->dev_enable_peer(0, num_devices);
  
  // ----------------------------------------------------------------

  // Device-device transfers

  printf("\nDevice-to-device (D2D) w/ peer enabled\n");

  pm->dev_profile_start("Device-to-device (P2P");
  
  for(int i=0; i<num_devices; ++i) {

    pm->dev_set_device(i);
    
    for(int j=0; j<num_devices; ++j) {

      if(i == j) {
	double t0 = MPI_Wtime();

	for(int k=0; k<_NUM_ITERATIONS; ++k) pm->dev_copy(d_b[j], d_a[i], size);

	double t = MPI_Wtime() - t0;

	char name[50];
	sprintf(name, "%i - %i  D2D copy w/ peer",i,i);
	
	print_summary(t, _N, _NUM_ITERATIONS, name);
      } else {
	double t0 = MPI_Wtime();

	for(int k=0; k<_NUM_ITERATIONS; ++k) pm->dev_memcpy_peer(d_b[j], j, d_a[i], i, size);

	double t = MPI_Wtime() - t0;
	
	char name[50];
	sprintf(name, "%i - %i  D2D transfer w/ peer",i,j);
	
	print_summary(t, _N, _NUM_ITERATIONS, name);
      }

      pm->dev_barrier();
      
    }
    printf("\n");
  }

  pm->dev_profile_stop();
  
  // Clean up

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    pm->dev_free(d_a[i]);
    pm->dev_free(d_b[i]);
  }
  
  pm->dev_free_host(d_a);
  pm->dev_free_host(d_b);
  pm->dev_free_host(a);
  
  delete pm;
  
  MPI_Finalize();
}
