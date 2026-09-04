/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose(double * out, double * in, int nrow, int ncol)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nrow; ++i)
    for(int j=0; j<ncol; ++j)
      out[j*nrow + i] = in[i*ncol + j];
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::vecadd(const double * in, double * out, int N)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<N; ++i) out[i] += in[i];
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::vecadd_batch(const double * in, double * out, int N, int num_batches)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<N; ++i) {
    double val = 0.0;
    for(int j=0; j<num_batches; ++j) val += in[j*N + i];
    out[i] += val;
  }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<N; ++i)
    for(int j=0; j<num_batches; ++j) inout[j*stride + offset + i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::set_to_zero(double * array, int size)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<size; ++i) array[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::veccopy(const double * src, double *dest, int size)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<size; ++i) dest[i] = src[i];
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_021(double * in, double * out, int ax1, int ax2, int ax3)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ax1; ++i)
    for(int j=0; j<ax2; ++j)
      for(int k=0; k<ax3; ++k) {
        int inputIndex = (i*ax3+k)*ax2+j;
        int outputIndex = (i*ax2+j)*ax3+k;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_102(double * in, double * out, int ax1, int ax2, int ax3)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ax1; ++i)
    for(int j=0; j<ax2; ++j)
      for(int k=0; k<ax3; ++k) {
        int inputIndex = (j*ax1+i)*ax3+k;
        int outputIndex = (i*ax2+j)*ax3+k;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int idx1=0; idx1<ax1; ++idx1)
    for(int idx2=0; idx2<ax2; ++idx2)
      for(int idx3=0; idx3<ax3; ++idx3)
        for(int idx4=0; idx4<ax4; ++idx4) {
          int outputIndex = ((idx3*ax2 + idx2)*ax4 + idx4)*ax1 + idx1;
          int inputIndex = ((idx1*ax2 + idx2)*ax3 + idx3)*ax4 + idx4;
          out[outputIndex] = in[inputIndex];
        }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_3210(double* in, double* out, int nmo, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ncas; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k)
        for(int l=0; l<nmo; ++l) {
          int inputIndex = ((i*ncas+j)*ncas+k)*nmo+l;
          int outputIndex = l*ncas*ncas*ncas + k*ncas*ncas + j*ncas + i;
          out[outputIndex] = in[inputIndex];
        }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_120(double * in, double * out, int naux, int nao, int ncas, int order)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<nao; ++k) {
        int inputIndex = i*nao*ncas + j*nao + k;
        int outputIndex = j*nao*naux + k*naux + i;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::getjk_unpack_buf2(double * buf2, double * eri, int * map, int naux, int nao, int nao_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i) {
    for(int j=0; j<nao; ++j) {
      double * buf = &(buf2[i*nao*nao]);
      double * tril = &(eri[i*nao_pair]);
      const int indx = j*nao;
      for(int k=0; k<nao; ++k) buf[indx+k] = tril[map[indx+k]];
    }
  }
}


#endif
