//Setup affinity of GPUs with cpus. 
//Affinity - Assigning software work (threads) to specific cpu cores or gpu
//This mini-app can be used to help check/confirm how software processes (threads) are assigned to compute hardware (cpu cores and gpus). 
//This is useful when debugging job launchers, like mpiexec, and ensuring all GPUs in a node are used as expected.
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>

#include <iomanip>
#include <vector>
#include <tuple>

#include <mpi.h>
#include <omp.h>

#include "pm/pm.h"

using namespace PM_NS;

// xthi.c from http://docs.cray.com/books/S-2496-4101/html-S-2496-4101/cnlexamples.html
// util-linux-2.13-pre7/schedutils/taskset.c
void get_cores(char *str)
{ 
  cpu_set_t mask;
  sched_getaffinity(0, sizeof(cpu_set_t), &mask);

  char *ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, &mask)) {
      int run = 0;
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &mask)) run++;
        else break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0) ptr++;
    }
  }
  ptr -= entry_made;
  *ptr = 0;
}

int main(int argc, char *argv[])
{
  // Initialize MPI
  
  int rnk, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm world = MPI_COMM_WORLD;
  
  MPI_Comm_rank(world, &rnk);
  MPI_Comm_size(world, &nprocs);

  class PM * pm = new PM();

  char nname[16];
  gethostname(nname, 16);
 
  // Initialize gpu

  int num_devices = pm->dev_num_devices();

  int num_threads = -1;
#pragma omp parallel
  num_threads = omp_get_num_threads();
  
  if(rnk == 0) printf("rnk= %i :  # of devices detected= %i  # of OpenMP threads= %i\n", rnk, num_devices, num_threads);
  MPI_Barrier(MPI_COMM_WORLD);
  
  for(int ir=0; ir<nprocs; ++ir) {

    if(ir == rnk) {

      char list_cores[7*CPU_SETSIZE];
      get_cores(list_cores);
      
      printf("To affinity and beyond!! nname= %s  rnk= %d  list_cores= (%s)  num_devices= %i\n\n",
	     nname, rnk, list_cores, num_devices);

      pm->dev_properties(num_devices);
      printf("\n");

      #pragma omp parallel for ordered
      for(int it=0; it<num_threads; ++it) {
        char list_cores[7*CPU_SETSIZE];
        get_cores(list_cores);
        #pragma omp ordered
        printf("To affinity and beyond!! nname= %s  rnk= %d  tid= %d: list_cores= (%s)\n",
                nname, rnk, omp_get_thread_num(), list_cores);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Finalize();
}
