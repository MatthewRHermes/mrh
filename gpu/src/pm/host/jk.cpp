/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

// DEPRECATED: legacy integral engine (fdrv helper) -- commented out. See refactor_plan.md.
#if 0
void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf)
{
  // not used by the host backend
}
#endif // end DEPRECATED Device::fdrv

/* ---------------------------------------------------------------------- */

/* Host translations of the CUDA kernels in device_cuda.cpp

/* ---------------------------------------------------------------------- */

void DeviceJk::getjk_rho(double * rho, double * dmtril, double * eri, int nset, int naux, int nao_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nset; ++i) {
    for(int j=0; j<naux; ++j) {
      double val = 0.0;
      for(int k=0; k<nao_pair; ++k)
        val += dmtril[i*nao_pair + k] * eri[j*nao_pair + k];
      rho[i*naux + j] = val;
    }
  }
}

/* ---------------------------------------------------------------------- */

void DeviceJk::getjk_vj(double * vj, double * rho, double * eri, int nset, int nao_pair, int naux, int init)
{
  const int gs_nao_pair = (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE;
  const int chunk_size = (gs_nao_pair <= _CUDA_MAX_GRID_DIM_YZ) ? gs_nao_pair : _CUDA_MAX_GRID_DIM_YZ;
  const int num_chunks = (gs_nao_pair <= _CUDA_MAX_GRID_DIM_YZ) ? 1 : (gs_nao_pair / _CUDA_MAX_GRID_DIM_YZ + 1);
  const int z_block = chunk_size * _DOT_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
  for(int i=0; i<nset; ++i) {
    for(int j=0; j<num_chunks; ++j) {
      for(int k=0; k<z_block; ++k) {
        int indxK = j*chunk_size + k;
        if(indxK >= nao_pair) continue;
        double val = 0.0;
        for(int l=0; l<naux; ++l) val += rho[i*naux + l] * eri[l*nao_pair + indxK];
        if(init) vj[i*nao_pair + indxK] = val;
        else vj[i*nao_pair + indxK] += val;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

#endif
