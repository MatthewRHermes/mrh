/* -*- c++ -*- */

#if defined(_GPU_HIP)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _TRANSPOSE_BLOCK_SIZE 16
#define _TRANSPOSE_NUM_ROWS 16
#define _UNPACK_BLOCK_SIZE 32
#define _HESSOP_BLOCK_SIZE 32
#define _DEFAULT_BLOCK_SIZE 32

#define _ATOMICADD
#define _ACCELERATE_KERNEL

#define _TILE(A,B) (A + B - 1) / B
#define _HIP_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* forward declarations of kernels defined in common.cpp */
__global__ void _vecadd(const double * in, double * out, int N);
__global__ void _veccopy(const double * src, double *dest, int size);

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCItrans_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(str0 >= na) return;
    if(j >= nlinka) return;
    int * tab  = &(link_index[4*nlinka*str0+4*j]);
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    int sign = tab[3];
    if (sign == 0) return;
    double * pket = &(ciket[str0*nb]);
    double * pbra = &(cibra[str1*nb]);
    for (int k=0; k<nb; ++k){
       atomicAdd(&(rdm[a*norb+i]), sign*pbra[k]*pket[k]);
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCItrans_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    if(str0 >= na) return;
    if(k >= nb) return;
    if(j >= nlinkb) return;
    double * pbra = &(cibra[str0*nb]);
    double tmp = ciket[str0*nb + k];
    int * tab  = &(link_index[4*nlinkb*k+4*j]);
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    int sign = tab[3];
    atomicAdd(&(rdm[a*norb + i]), sign*pbra[str1]*tmp);
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCItrans_rdm1a_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinka, 
                                            int ia_ket, int ja_ket, int ib_ket, int jb_ket, 
                                            int ia_bra, int ja_bra, int ib_bra, int jb_bra, 
                                            int na_bra, int nb_bra, int na_ket, int nb_ket, 
                                            int b_len, int b_bra_offset, int b_ket_offset, 
                                            int sign_dummy, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //if(str0 >= na) return;
    if(str0 >= na_ket) return;//ciket is 0 if k is outside ia_bra:ja_bra
    if(j >= nlinka) return;
    int * tab  = &(link_index[4*nlinka*(str0+ia_ket)+4*j]);
    int sign = tab[3];
    if (sign == 0) return;
    sign = sign * sign_dummy; 
    int str1 = tab[2];
    if ((str1>=ia_bra) && (str1<ja_bra)){
      int a = tab[0];
      int i = tab[1];
      //double * pket = &(ciket[str0*nb]);
      double * pket = &(ciket[str0*nb_ket]);
      //double * pbra = &(cibra[str1*nb]);
      double * pbra = &(cibra[(str1-ia_bra)*nb_bra]);
      //for (int k=0; k<nb; ++k){
      for (int k=0; k<b_len; ++k){ // only from  max(ib_bra, ib_ket): min(jb_bra, jb_ket)
         //atomicAdd(&(rdm[a*norb+i]), sign*pbra[k-b_bra_offset-ib_bra]*pket[k-b_ket_offset-ib_ket]);
         atomicAdd(&(rdm[a*norb+i]), sign*pbra[k+b_bra_offset]*pket[k+b_ket_offset]);
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCItrans_rdm1b_v2( double * cibra, double * ciket, double * rdm, int norb, int nlinkb, 
                                            int ia_ket, int ja_ket, int ib_ket, int jb_ket, 
                                            int ia_bra, int ja_bra, int ib_bra, int jb_bra, 
                                            int na_bra, int nb_bra, int na_ket, int nb_ket, 
                                            int a_len, int ia_max, 
                                            int sign_dummy, int * link_index)
{
  int str0 = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.z * blockDim.z + threadIdx.z;
  //if(str0 >= na) return;
  if(str0 >= a_len) return;//ci[str0*nb] accessed for both, ia_max < str0 < ja_min, a_len = ja_min - ia_max
  //if(k >= nb) return;
  if(k >= nb_ket) return;
  if(j >= nlinkb) return;
  //double * pbra = &(cibra[str0*nb]);
  double * pbra = &(cibra[(str0+ia_max)*nb_bra]);
  //double tmp = ciket[str0*nb + k];
  double tmp = ciket[(str0+ia_max)*nb_ket + k];
  //int * tab  = &(link_index[4*nlinkb*k+4*j]);
  int * tab  = &(link_index[4*nlinkb*(k+ib_ket)+4*j]);
  int str1 = tab[2];
  if ((str1>=ib_bra)&&(str1<jb_bra)){
    int sign = tab[3];
    if (sign ==0 ) return;
      sign = sign*sign_dummy;
      int a = tab[0];
      int i = tab[1];
      atomicAdd(&(rdm[a*norb + i]), sign*pbra[str1-ib_bra]*tmp);
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCImake_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0>=na) return ;
    if (j>=nlinka) return ;
    double * pci0 = &(ciket[str0*nb]);
    #ifdef _ACCELERATE_KERNEL 
    int * tab = &(link_index[4*nlinka*str0 + 4*j]); 
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    int sign = tab[3];
    #else
    int a = link_index[4*nlinka*str0 + 4*j]; 
    int i = link_index[4*nlinka*str0 + 4*j + 1]; 
    int str1 = link_index[4*nlinka*str0 + 4*j + 2]; 
    int sign = link_index[4*nlinka*str0 + 4*j + 3];
    #endif

    double * pci1 = &(ciket[str1*nb]);
    if (a>=i && sign!=0){
      for (int k=0;k<nb; ++k){
        atomicAdd(&(rdm[a*norb+i]), sign*pci0[k]*pci1[k]);
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCImake_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;
    if (str0>=na) return ;
    if (k>=nb) return ;
    if (j>=nlinkb) return ;
    double * pci0 = &(ciket[str0*nb]);
    #ifdef _ACCELERATE_KERNEL
    int * tab = &(link_index[4*nlinkb*k + 4*j]); 
    int a = tab[0];
    int i = tab[1];
    int sign = tab[3];
    if (a>=i && sign!=0) { 
    int str1 = tab[2];
    atomicAdd(&(rdm[a*norb+i]), sign*pci0[str1]*pci0[k]);
      }
    #else
    int a = link_index[4*nlinkb*k + 4*j]; 
    int i = link_index[4*nlinkb*k + 4*j + 1]; 
    int str1 = link_index[4*nlinkb*k + 4*j + 2]; 
    int sign = link_index[4*nlinkb*k + 4*j + 3];
    if (a>=i && sign!=0) { 
    atomicAdd(&(rdm[a*norb+i]), sign*pci0[str1]*pci0[k]);
      }
    #endif
}

/* ---------------------------------------------------------------------- */

__global__ void _symmetrize_rdm(int norb, double * rdm)
{
  for (int i=0; i<norb; ++i){
    for (int j=0; j<i; ++j){
        rdm[j*norb+i] = rdm[i*norb+j];
      }
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_a_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int * link_index)
{
    //this works.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= nlinka) return;
    if (k >= nb) return;
    int norb2 = norb*norb;
    #ifdef _ACCELERATE_KERNEL 
    int * tab = &(link_index[4*nlinka*stra_id + 4*j]); 
    int sign = tab[3];
    if (sign == 0) return;
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    atomicAdd(&(buf[k*norb2 + i*norb + a]), sign*ci[str1*nb + k]);
    
    #else
    #ifdef _DEBUG_ATOMICADD
    atomicAdd(&(buf[k*norb2 + i*norb + a]), sign*ci[str1*nb + k]);
    #else
    buf[k*norb2 + i*norb + a] += sign*ci[str1*nb + k];
    #endif
    #endif
    //TODO: implement csum 
    // Is it necessary to? 
    // Sure, in case when it's blocked over nb of size 100 determinants at once, 
    // but over entire nb, do you think it will be 0 enough times to get the performance boost?
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_b_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinkb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0 >= nb) return;
    if (j >= nlinkb) return;
    int norb2 = norb*norb;
    //tab = clink_indexb + strb_id*nlinkb // remember strb_id = 0 since we are doing the entire b at once
    //for (str0<nb) {for (j<nb) {t1[i*norb+a] += sign * pci[str1];} t1+=norb2; tab+=nlinkb;}
    #ifdef _ACCELERATE_KERNEL
    int * tab = &(link_index[4*str0*nlinkb+4*j]);
    int sign = tab[3];
    if (sign==0) return;
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[stra_id*nb + str1]);
    #else
    int a = link_index[4*str0*nlinkb + 4*j]; 
    int i = link_index[4*str0*nlinkb + 4*j + 1]; 
    int str1 = link_index[4*str0*nlinkb + 4*j + 2]; 
    int sign = link_index[4*str0*nlinkb + 4*j + 3];
      #ifdef _DEBUG_ATOMICADD
      atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[stra_id*nb + str1]);
      #else
      buf[str0*norb2 + i*norb + a] += sign*ci[stra_id*nb+str1];
      #endif
    #endif
    //TODO: implement csum 
    // Refer to comment in _compute_FCIrdm2_a_t1ci 
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_a_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int * link_index)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int k = blockIdx.y * blockDim.y + threadIdx.y;
    //if (j >= nlinka) return;
    int norb2 = norb*norb;
    if (k >= nb) return;
    int * tab_line = &(link_index[4*nlinka*stra_id]); 
   
    for (int j=0;j<nlinka;++j){
    int * tab = &(tab_line[4*j]);
    int sign = tab[3];
    if (sign != 0){
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    //atomicAdd(&(buf[k*norb2 + i*norb + a]), sign*ci[str1*nb + k]);
    buf[k*norb2 + i*norb + a]+= sign*ci[str1*nb + k];}
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_b_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinkb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0 >= nb) return;
    //if (j >= nlinkb) return;
    int norb2 = norb*norb;
    int * tab_line = &(link_index[4*str0*nlinkb]); 
    for (int j=0;j<nlinkb;++j){
    int * tab = &(tab_line[4*j]);
    int sign = tab[3];
    if (sign!=0){
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    //atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[stra_id*nb + str1]);
    buf[str0*norb2 + i*norb + a] += sign*ci[stra_id*nb + str1];}
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_a_t1ci_v3(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int * link_index)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int norb2 = norb*norb;
    if (k >= nb) return;
    int * tab_line = &(link_index[4*nlinka*stra_id]); 

    double * tmp_buf = &(buf[k*norb2]);
    for (int j=threadIdx.y;j<nlinka;j+=blockDim.y){
    int * tab = &(tab_line[4*j]);
    int sign = tab[3];
    if (sign != 0){
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    tmp_buf[i*norb + a]+= sign*ci[str1*nb + k];}
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_b_t1ci_v3(double * ci, double * buf, int stra_id, int nb, int norb, int nlinkb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0 >= nb) return;
    //if (j >= nlinkb) return;
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[str0*norb2]);
    int * tab_line = &(link_index[4*str0*nlinkb]); 
    for (int j=threadIdx.y;j<nlinkb;j+=blockDim.y){
    int * tab = &(tab_line[4*j]);
    int sign = tab[3];
    if (sign!=0){
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    //atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[stra_id*nb + str1]);
    tmp_buf[i*norb + a] += sign*ci[stra_id*nb + str1];}
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_a_t1ci_v4(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinka, int * link_index)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_id >= batches) return;
    if (k >= nb) return;
    int norb2 = norb*norb;
    int * tab_line = &(link_index[4*nlinka*(stra_id+batch_id)]); 

    double * tmp_buf = &(buf[batch_id*norb2*nb+k*norb2]);
    for (int j=threadIdx.z;j<nlinka;j+=blockDim.z){
    int * tab = &(tab_line[4*j]);
    int sign = tab[3];
    if (sign != 0){
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    tmp_buf[i*norb + a]+= sign*ci[str1*nb + k];}
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm2_b_t1ci_v4(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinkb, int * link_index)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int str0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_id >= batches) return;
    if (str0 >= nb) return;
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[batch_id*norb2*nb + str0*norb2]);
    int * tab_line = &(link_index[4*str0*nlinkb]); 
    double * tmp_ci = &(ci[(stra_id+batch_id)*nb]);
    for (int j=threadIdx.z;j<nlinkb;j+=blockDim.z){
    int * tab = &(tab_line[4*j]);
    int sign = tab[3];
    if (sign!=0){
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    //atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[stra_id*nb + str1]);
    tmp_buf[i*norb + a] += sign*tmp_ci[str1];}
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_a_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nlinka) return;
    int norb2 = norb*norb;
    int * tab = &(link_index[4*nlinka*stra_id + 4*j]); 
    //for (int k=ib; k<jb; ++k){//k is the beta loop
    for (int k=0; k<jb-ib; ++k){// Doing this because ci[:, ib:jb] is filled, rest is zeros.
                                // Also, buf only needs to get populated from ib<k<jb, so less data needs to be added
      int sign = tab[3];
      if (sign != 0) {
        int str1 = tab[2];
        if ((str1>=ia) && (str1<ja)){//str1 is alpha loop
          int a = tab[0];
          int i = tab[1];
          atomicAdd(&(buf[(k+ib)*norb2 + i*norb + a]), sign*ci[(str1-ia)*nb + k]);//I'm not sure how this plays out in the bigger kernel, so keeping as k+ib on the buf side
          }
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_b_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0 >= nb) return;
    if (j >= nlinkb) return;
    int norb2 = norb*norb;
    int * tab = &(link_index[4*str0*nlinkb+4*j]);
    int sign = tab[3];
    if (sign!=0){ //return;
      int str1 = tab[2];
      if ((str1>=ib) && (str1<jb)){
        int a = tab[0];
        int i = tab[1];
        atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[(stra_id-ia)*nb_bra + str1-ib]);// rdm3h_b_t1ci is only called when stra_id is more than ia
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_a_t1ci_v3(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    //if (j >= nlinka) return;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= jb-ib) return;
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[(k+ib)*norb2]);
    //int * tab = &(link_index[4*nlinka*stra_id + 4*j]); 
    int * tab_line = &(link_index[4*nlinka*stra_id]); 
    //for (int k=0; k<jb-ib; ++k){// Doing this because ci[:, ib:jb] is filled, rest is zeros.
    for (int j=0; j<nlinka; ++j){
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if (sign != 0) {
        int str1 = tab[2];
        if ((str1>=ia) && (str1<ja)){
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(str1-ia)*nb + k];//I'm not sure how this plays out in the bigger kernel, so keeping as k+ib on the buf side
          }
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_b_t1ci_v3(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0 >= nb) return;
    //if (j >= nlinkb) return;
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[str0*norb2]);
    //int * tab = &(link_index[4*str0*nlinkb+4*j]);
    int * tab_line = &(link_index[4*str0*nlinkb]);
    for (int j=0;j<nlinkb;++j){
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if (sign!=0){ //return;
        int str1 = tab[2];
        if ((str1>=ib) && (str1<jb)){
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(stra_id-ia)*nb_bra + str1-ib];// rdm3h_b_t1ci is only called when stra_id is more than ia
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_a_t1ci_v4(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    //if (j >= nlinka) return;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= jb-ib) return;
     
    //int na = ja - ia;for transpose version if we ever do it
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[(k+ib)*norb2]);
    int * tab_line = &(link_index[4*nlinka*stra_id]); 
    for (int j=threadIdx.y; j<nlinka; j+=blockDim.y){
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if (sign != 0) {
        int str1 = tab[2];
        if ((str1>=ia) && (str1<ja)){
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(str1-ia)*nb + k];//
          //!!!this is incorrect, just doing this for checking speedups because then the data is accessed contiguously, and hopefully fewer cache misses
          //tmp_buf[i*norb + a] += sign*ci[k*na+(str1-ia)];//speedup is 3%, revisit this later. 
          }
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_b_t1ci_v4(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (str0 >= nb) return;
    //if (j >= nlinkb) return;
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[str0*norb2]);
    //int * tab = &(link_index[4*str0*nlinkb+4*j]);
    int * tab_line = &(link_index[4*str0*nlinkb]);
    for (int j=threadIdx.y;j<nlinkb; j += blockDim.y){
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if (sign!=0){ //return;
        int str1 = tab[2];
        if ((str1>=ib) && (str1<jb)){
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(stra_id-ia)*nb_bra + str1-ib];// rdm3h_b_t1ci is only called when stra_id is more than ia
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_a_t1ci_v5(double * ci, double * buf, int stra_id, int batches, int nb, int nb_ci, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_id >= batches) return;
    if (k >= jb-ib) return;
    //printf("batch_id: %i k: %i ib: %i\n",batch_id, k, ib); 
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[batch_id*norb2*nb + (k+ib)*norb2]);
    int * tab_line = &(link_index[4*nlinka*(stra_id+batch_id)]); 
    for (int j=threadIdx.z; j<nlinka; j+=blockDim.z){
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if (sign != 0) {
        int str1 = tab[2];
        if ((str1>=ia) && (str1<ja)){
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(str1-ia)*nb_ci + k];//
          }
        }
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _compute_FCIrdm3h_b_t1ci_v5(double * ci, double * buf, int stra_id, int batches, int nb, int nb_ci, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int str0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_id >= batches) return;
    if (str0 >= nb) return;
    int norb2 = norb*norb;
    double * tmp_buf = &(buf[batch_id*norb2*nb + str0*norb2]);
    //int * tab = &(link_index[4*str0*nlinkb+4*j]);
    int * tab_line = &(link_index[4*str0*nlinkb]);
    for (int j=threadIdx.z;j<nlinkb; j += blockDim.z){
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if (sign!=0){ //return;
        int str1 = tab[2];
        if ((str1>=ib) && (str1<jb)){
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(stra_id+batch_id-ia)*nb_ci + str1-ib];// rdm3h_b_t1ci is only called when stra_id is more than ia
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_jikl(const double * in, double *out, int norb)
{
    int norb2 = norb*norb;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= norb2) return;
    for (int i =0; i<norb; ++i){ 
        for (int j=0; j<norb; ++j){
          const double * tmp_in = &(in[(i*norb+j)*norb2]); 
          double * tmp_out = &(out[(j*norb+i)*norb2]); 
          tmp_out[k] = tmp_in[k];
        } 
      }
}

/* ---------------------------------------------------------------------- */

__global__ void _add_rdm1_to_2(double * dm1, double * dm2, int norb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=norb) return;
    if (j>=norb) return;
    if (k>=norb) return;
    //double * tmp_rdm2 = &(dm2[((i*norb+j)*norb+j)*norb + k]);
    //double * tmp_rdm1 = &(dm1[i*norb + k]);
    //printf("i:%i j:%i k:%i dm1loc: %i dm2loc: %i dm1: %f dm2: %f\n",i,j,k,i*norb + k, ((i*norb+j)*norb+j)*norb + k, dm1[i*norb + k], dm2[((i*norb+j)*norb+j)*norb + k]);
    dm2[((i*norb+j)*norb+j)*norb + k] -= dm1[i*norb + k];
}

/* ---------------------------------------------------------------------- */

__global__ void _add_rdm_transpose(double * buf, double * dm2, int norb)
{
    int norb2 = norb*norb;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=norb2) return;
    if (j>=norb2) return;
    buf[i*norb2 + j] += dm2[j*norb2+i];
}

/* ---------------------------------------------------------------------- */

__global__ void _build_rdm(double * buf, double * dm2, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dm2[i] = buf[i]/2;
}

/* ---------------------------------------------------------------------- */

__global__ void _filter_sfudm(const double * dm2, double * dm1, int norb)
{
    //already passing in the pointer to dm2[-1, :, :, :] 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= norb) return;
    if (j >= norb) return;
    int norb1 = norb+1;
    int norb12 = (norb+1)*(norb+1);
    dm1[i*norb+j] = dm2[i*norb12+j*norb1+norb];
}

/* ---------------------------------------------------------------------- */

__global__ void _filter_tdmpp(const double * dm2, double * dm1, int norb, int spin)
{
    //only need dm2[:-ndum,-1,:-ndum,-ndum] //ndum = 2-(spin%2)
    //norb includes ndum
    int ndum = (spin!=1) ? 2:1; 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= norb-ndum) return;
    if (j >= norb-ndum) return;
    dm1[i*(norb-ndum)+j] = dm2[i*norb*norb*norb + (norb-1)*norb*norb + j*norb+ norb-ndum];
}

/* ---------------------------------------------------------------------- */

__global__ void _filter_tdm1h(const double * in, double * out, int norb)
{

    //tdm1h = tdm1h.T
    //tdm1h = tdm1h[-1,:-1]
    //in is (norb+1)^2
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= norb) return;
    out[i] = in[i*(norb+1)+norb];
}

/* ---------------------------------------------------------------------- */

__global__ void _filter_tdm3h(double * in, double * out, int norb)
{
    //tdm3h = tdm3h[:-1,-1,:-1,:-1]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= norb) return;
    if (j >= norb) return;
    if (k >= norb) return;
    int norb1 = norb+1;
    //printf("%i %i %i %i %f\n",i, j, k, ((i*norb1+norb)*norb1+j)*norb1+k, in[((i*norb1+norb)*norb1+j)*norb1+k]);
    out[(i*norb+j)*norb+k] = in[((i*norb1+norb)*norb1+j)*norb1+k];
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCItrans_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nlinka,block_size.y),1);

  hipStream_t s = *(ctx.pm->dev_get_queue());

  _compute_FCItrans_rdm1a<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinka, link_index);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rdm_from_ci; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCItrans_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nb,block_size.y),_TILE(nlinkb, block_size.z));
  
  hipStream_t s = *(ctx.pm->dev_get_queue());


  _compute_FCItrans_rdm1b<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinkb, link_index);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rdm_from_ci; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCItrans_rdm1a_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinka, 
                                        int ia_bra, int ja_bra, int ib_bra, int jb_bra, 
                                        int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sign, 
                                        int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  int na_bra = ja_bra - ia_bra; 
  int na_ket = ja_ket - ia_ket; 
  int nb_bra = jb_bra - ib_bra; 
  int nb_ket = jb_ket - ib_ket; 
  int ib_max = (ib_bra > ib_ket) ? ib_bra : ib_ket;
  int jb_min = (jb_bra < jb_ket) ? jb_bra : jb_ket;
  int b_len  = jb_min - ib_max;
  if (b_len>0){
    int b_bra_offset = ib_max - ib_bra;
    int b_ket_offset = ib_max - ib_ket;

    dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
    dim3 grid_size(_TILE(na_ket, block_size.x),_TILE(nlinka,block_size.y),1);

    _compute_FCItrans_rdm1a_v2<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, nlinka, 
                                                             ia_ket, ja_ket, ib_ket, jb_ket, 
                                                             ia_bra, ja_bra, ib_bra, jb_bra, 
                                                             na_bra, nb_bra, na_ket, nb_ket, 
                                                             b_len, b_bra_offset, b_ket_offset, 
                                                             sign, link_index);
    }
#ifdef _DEBUG_DEVICE
    printf("na_ket: %i ia_ket: %i ja_ket: %i ib_ket: %i ib_bra: %i nb_bra: %i nb_ket: %i b_len: %i b_bra_offset: %i b_ket_offset: %i sign: %i\n",na_ket, ia_ket, ja_ket, ib_ket, ib_bra, nb_bra, nb_ket, b_len, b_bra_offset, b_ket_offset, sign);
#endif
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCItrans_rdm1b_v2( double * cibra, double * ciket, double * rdm, int norb, int nlinkb, 
                                        int ia_bra, int ja_bra, int ib_bra, int jb_bra, 
                                        int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sign, 
                                        int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());

  int na_bra = ja_bra - ia_bra; 
  int na_ket = ja_ket - ia_ket; 
  int nb_bra = jb_bra - ib_bra; 
  int nb_ket = jb_ket - ib_ket; 
  int ia_max = (ia_bra > ia_ket) ? ia_bra : ia_ket;
  int ja_min = (ja_bra < ja_ket) ? ja_bra : ja_ket;
  int a_len  = ja_min - ia_max;
  if (a_len>0){
    //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
    dim3 block_size(1,_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE);
    dim3 grid_size(_TILE(a_len, block_size.x),_TILE(nb_ket,block_size.y),_TILE(nlinkb, block_size.z));
  
    _compute_FCItrans_rdm1b_v2<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, nlinkb, 
                                                             ia_ket, ja_ket, ib_ket, jb_ket, 
                                                             ia_bra, ja_bra, ib_bra, jb_bra, 
                                                             na_bra, nb_bra, na_ket, nb_ket, 
                                                             a_len, ia_max, 
                                                             sign, link_index);
    }
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCImake_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  {dim3 block_size(_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE,1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nlinka, block_size.y),1);
  _compute_FCImake_rdm1a<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinka, link_index);
  #ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rdm_from_ci; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
  #endif
  }
  {dim3 block_size (1,1,1);
   dim3 grid_size(1,1,1);
   _symmetrize_rdm<<<grid_size, block_size, 0, s>>> (norb, rdm);}
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCImake_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  {
  //dim3 block_size(_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE); //TODO: fix this?
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nb, block_size.y),_TILE(nlinkb, block_size.z));
  #ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_rdm1b; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  #endif
  _compute_FCImake_rdm1b<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinkb, link_index);
  _HIP_CHECK_ERRORS();
  }
  {dim3 block_size(1,1,1);
   dim3 grid_size(1,1,1); 
   _symmetrize_rdm<<<grid_size, block_size, 0, s>>> (norb, rdm); }
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCIrdm2_a_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinka, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches, block_size.x),_TILE(nb, block_size.y), 1);
  _compute_FCIrdm2_a_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id,batches,nb, norb, nlinka, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCIrdm2_b_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinkb, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  {dim3 block_size(1, 1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches,block_size.x),_TILE(nb, block_size.y), 1);
  _compute_FCIrdm2_b_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, batches, nb, norb, nlinkb, link_index);}
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_b_t1ci; :: Nb= %i Norb =%i Nlinkb =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinkb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCIrdm3h_a_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  #if 0
  dim3 block_size(_DEFAULT_BLOCK_SIZE,1,1);
  dim3 grid_size(_TILE(jb-ib, block_size.x), 1, 1);
  _compute_FCIrdm3h_a_t1ci_v3<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinka, ia, ja, ib, jb, link_index);
  #else
  dim3 block_size(1,_DEFAULT_BLOCK_SIZE,1);
  dim3 grid_size(_TILE(jb-ib, block_size.x), 1, 1);
  _compute_FCIrdm3h_a_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinka, ia, ja, ib, jb, link_index);
  #endif
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCIrdm3h_b_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  #if 0
  dim3 block_size(_DEFAULT_BLOCK_SIZE,1,1);
  dim3 grid_size(_TILE(nb, block_size.x), 1, 1);
  _compute_FCIrdm3h_b_t1ci_v3<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, nb_bra, norb, nlinkb, ia, ja, ib, jb, link_index);
  #else
  dim3 block_size(1,_DEFAULT_BLOCK_SIZE,1);
  dim3 grid_size(_TILE(nb, block_size.x), 1, 1);
  _compute_FCIrdm3h_b_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, nb_bra, norb, nlinkb, ia, ja, ib, jb, link_index);
  #endif
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_b_t1ci; :: Nb= %i Norb =%i Nlinkb =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinkb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCIrdm3h_a_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_ci, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(1, 1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches,block_size.x), _TILE(jb-ib, block_size.y), 1);
  _compute_FCIrdm3h_a_t1ci_v5<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, batches, nb, nb_ci, norb, nlinka, ia, ja, ib, jb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCIrdm3h_b_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(1, 1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches, block_size.x), _TILE(nb, block_size.y), 1);
  _compute_FCIrdm3h_b_t1ci_v5<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, batches, nb, nb_bra, norb, nlinkb, ia, ja, ib, jb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_b_t1ci; :: Nb= %i Norb =%i Nlinkb =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinkb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::transpose_jikl(double * tdm, double * buf, int norb)
{
  int norb2 = norb*norb;
  hipStream_t s = *(ctx.pm->dev_get_queue());
  {
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1,1); 
  dim3 grid_size(_TILE(norb2, block_size.x), 1,1);
  _transpose_jikl<<<grid_size,block_size,0,s>>>(tdm,buf,norb);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::transpose_jikl; :: Norb= %i grid_size= %i %i %i  block_size= %i %i %i\n",
	 norb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif  
  }
  ctx.utils->veccopy(buf, tdm, norb2*norb2);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::reduce_buf3_to_rdm(const double * buf3, double * dm2, int size_tdm2, int num_gemm_batches)
{
  ctx.utils->vecadd_batch(buf3, dm2, size_tdm2, num_gemm_batches);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::reorder(double * dm1, double * dm2, double * buf, int norb)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  //for k in range (norb): rdm2[:,k,k,:] -= rdm1.T //remember, rdm1 is returned as rdm1.T, so double transpose, hence just rdm1
  
  dim3 block_size (1,1,1);
  dim3 grid_size (_TILE(norb, block_size.x), _TILE(norb, block_size.y), _TILE(norb, block_size.z));
  _add_rdm1_to_2<<<grid_size, block_size, 0, s>>> (dm1, dm2, norb);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_sfudm( const double * dm2, double * dm1, int norb)
{
  //only need dm2[-1,:-1, :-1, -1]
  hipStream_t s = *(ctx.pm->dev_get_queue());
  int norb_m1 = norb-1;
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(norb_m1, block_size.x),_TILE(norb_m1, block_size.y),1);
  _filter_sfudm<<<grid_size, block_size, 0,s>>>(dm2,dm1,norb_m1);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_tdmpp( const double * dm2, double * dm1, int norb, int spin)
{
  //only need dm2[:-ndum,-1,:-ndum,-ndum] //ndum = 2-(spin%2)
  int ndum = (spin!=1) ? 2:1; 
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(norb-ndum, block_size.x),_TILE(norb-ndum, block_size.y),1);
  _filter_tdmpp<<<grid_size, block_size, 0,s>>>(dm2,dm1,norb,spin);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_tdm1h( const double * in, double * out, int norb)
{
  //tdm1h = tdm1h.T
  //tdm1h = tdm1h[-1,:-1]
  //in is (norb+1)^2
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(norb, block_size.x),1,1);
  _filter_tdm1h<<<grid_size, block_size, 0,s>>>(in,out,norb);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_tdm3h(double * in, double * out, int norb)
{
  //tdm3h = tdm3h[:-1,-1,:-1,:-1]
  //dm2 is (norb+1)^4
  hipStream_t s = *(ctx.pm->dev_get_queue());
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(norb, block_size.x),_TILE(norb, block_size.y),_TILE(norb, block_size.z));
  _filter_tdm3h<<<grid_size, block_size, 0,s>>>(in, out,norb);
  _HIP_CHECK_ERRORS();
}


#endif
