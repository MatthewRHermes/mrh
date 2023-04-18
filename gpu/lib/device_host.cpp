/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "device.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

double Device::compute(double * data)
{ 
  // do something useful
  
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<n; ++i) {
    sum += data[i];
    data[i] += 1.0;
  }
    
  printf(" C-Kernel : n= %i  sum= %f\n",n, sum);
  
  return sum;
}

#endif
