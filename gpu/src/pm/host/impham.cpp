/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

#define _ATOMICADD
#define _ACCELERATE_KERNEL

//#define _DEBUG_DEVICE
//#define _DEBUG_H2EFF
//#define _DEBUG_H2EFF2
//#define _DEBUG_H2EFF_DF
//#define _DEBUG_AO2MO

#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

void DeviceImpham::pack_eri(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i) {
    for(int j=0; j<nao; ++j) {
      double * buf = &(buf2[i*nao*nao]);
      double * tril = &(eri1[i*nao_pair]);
      const int indx = j*nao;
      for(int k=0; k<nao; ++k) tril[map[indx+k]] = buf[indx+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

#endif
