/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device_pdft.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void DevicePdft::get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<ngrid; ++i) Pi[i] += rho[i]*rho[i];
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_gridkern(double * mo_grid, double * gridkern, int ngrid, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ngrid; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k) {
        double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
        double * tmp_mo_grid = &(mo_grid[i*ncas]);
        tmp_gridkern[j*ncas+k] = tmp_mo_grid[j]*tmp_mo_grid[k];
      }
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_buf_pdft(double * gridkern, double * buf, double * cascm2, int ngrid, int ncas)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<ngrid; ++i)
    for(int j=0; j<ncas*ncas; ++j) {
      double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
      double * tmp_cascm2 = &(cascm2[j*ncas*ncas]);
      double * tmp_out = &(buf[i*ncas*ncas+j]);
      for(int k=0; k<ncas*ncas; ++k) tmp_out[0] += tmp_gridkern[k]*tmp_cascm2[k];
    }
}

/* ---------------------------------------------------------------------- */

void DevicePdft::make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<ngrid; ++i) {
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_buf = &(buf[i*ncas*ncas]);
    double * tmp_Pi = &(Pi[i]);
    for(int j=0; j<ncas*ncas; ++j) tmp_Pi[0] += tmp_gridkern[j]*tmp_buf[j];
  }
}


#endif
