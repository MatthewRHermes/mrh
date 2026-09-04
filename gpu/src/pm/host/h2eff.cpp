/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void DeviceH2eff::extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<ncas; ++i)
    for(int j=0; j<ncas; ++j)
      small_mat[i*ncas + j] = big_mat[(i+ncore)*nmo + (j+ncore)];
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nmo*ncas; ++i)
    for(int j=0; j<ncas*ncas; ++j) {
      double * in_buf = &(in[i*ncas_pair]);
      double * out_buf = &(out[i*ncas*ncas]);
      out_buf[j] = in_buf[map[j]];
    }
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::transpose_2310(double * in, double * out, int nmo, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<nmo; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k)
        for(int l=0; l<ncas; ++l) {
          int inputIndex = ((i*ncas+j)*ncas+k)*ncas+l;
          int outputIndex = k*ncas*ncas*nmo + l*ncas*nmo + j*nmo + i;
          out[outputIndex] = in[inputIndex];
        }
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<nao; ++k) {
        int inputIndex = i*nao*ncas + j*nao + k;
        int outputIndex = k*ncas*naux + j*naux + i;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<nmo; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas_pair; ++k) {
        double * out_buf = &(out[(i*ncas + j)*ncas_pair]);
        double * in_buf = &(in[(i*ncas + j)*ncas*ncas]);
        out_buf[k] = in_buf[map[k]];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nmo*ncas; ++i)
    for(int j=0; j<ncas*ncas; ++j)
      out[i*ncas_pair + map[j]] = in[j*ncas*nmo + i];
}

/* ---------------------------------------------------------------------- */

void DeviceH2eff::pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<nmo*ncas; ++i)
    for(int j=0; j<ncas*ncas; ++j)
      out[i*ncas_pair + map[j]] += in[j*ncas*nmo + i];
}


#endif
