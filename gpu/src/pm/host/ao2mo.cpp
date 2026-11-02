/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufpa(const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<nmo; ++j)
      for(int k=0; k<ncas; ++k) {
        int inputIndex = (i*nmo + j)*nmo + k+ncore;
        int outputIndex = (i*nmo + j)*ncas + k;
        bufpa[outputIndex] = bufpp[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufaa(const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k) {
        int inputIndex = (i*nmo + (j+ncore))*nmo + k+ncore;
        int outputIndex = (i*ncas + j)*ncas + k;
        bufaa[outputIndex] = bufpp[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<ncas; ++i)
    for(int j=0; j<nao; ++j)
      small_mat[i*nao + j] = big_mat[j*nmo + i+ncore];
}

/* ---------------------------------------------------------------------- */

void DeviceAo2mo::get_bufd(const double* bufpp, double* bufd, int naux, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<nmo; ++j)
      bufd[i*nmo + j] = bufpp[(i*nmo + j)*nmo + j];
}


#endif
