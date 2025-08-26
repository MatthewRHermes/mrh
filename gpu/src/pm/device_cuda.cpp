/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "../device.h"

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

/* ---------------------------------------------------------------------- */

void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf) // this needs to be removed when host+sycl backends ready
{
//   const int ij_pair = nao * nao;
//   const int nao2 = nao * (nao + 1) / 2;
    
// #pragma omp parallel for
//   for (int i = 0; i < nij; i++) {
//     double * buf = &(_buf[i * nao * nao]);

//     int _i, _j, _ij;
//     double * tril = vin + nao2*i;
//     for (_ij = 0, _i = 0; _i < nao; _i++) 
//       for (_j = 0; _j <= _i; _j++, _ij++) {
// 	buf[_i*nao+_j] = tril[_ij];
// 	buf[_i+nao*_j] = tril[_ij]; // because going to use batched dgemm call on gpu
//       }
//   }
  
// #pragma omp parallel for
//   for (int i = 0; i < nij; i++) {
//     double * buf = &(_buf[i * nao * nao]);
    
//     const double D0 = 0;
//     const double D1 = 1;
//     const char SIDE_L = 'L';
//     const char UPLO_U = 'U';

//     double * _vout = vout + ij_pair*i;
    
//     dsymm_(&SIDE_L, &UPLO_U, &nao, &nao, &D1, buf, &nao, mo_coeff, &nao, &D0, _vout, &nao);    
//   }
  
}

/* ---------------------------------------------------------------------- */
/* CUDA kernels
/* ---------------------------------------------------------------------- */

#if 1
__global__ void _getjk_rho(double * rho, double * dmtril, double * eri1, int nset, int naux, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nset) return;
  if(j >= naux) return;

  __shared__ double cache[_RHO_BLOCK_SIZE];

  int k = blockIdx.z * blockDim.z + threadIdx.z;
  int cache_id = threadIdx.z;

  // thread-local work

  const int indxi = i * nao_pair;
  const int indxj = j * nao_pair;
  
  double tmp = 0.0;
  while (k < nao_pair) {
    tmp += dmtril[indxi + k] * eri1[indxj + k];
    k += blockDim.z; // * gridDim.z; // gridDim.z is just 1
  }

  cache[cache_id] = tmp;

  // block

  __syncthreads();

  // manually reduce values from threads within block

  int l = blockDim.z / 2;
  while (l != 0) {
    if(cache_id < l)
      cache[cache_id] += cache[cache_id + l];

    __syncthreads();
    l /= 2;
  }

  // store result in global array
  
  if(cache_id == 0)
    rho[i * naux + j] = cache[0];
}

#else

__global__ void _getjk_rho(double * rho, double * dmtril, double * eri1, int nset, int naux, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nset) return;
  if(j >= naux) return;

  double val = 0.0;
  for(int k=0; k<nao_pair; ++k) val += dmtril[i * nao_pair + k] * eri1[j * nao_pair + k];
  
  rho[i * naux + j] = val;
}
#endif

/* ---------------------------------------------------------------------- */

__global__ void _getjk_vj(double * vj, double * rho, double * eri1, int nset, int nao_pair, int naux, int init)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nset) return;
  if(j >= nao_pair) return;

  double val = 0.0;
  for(int k=0; k<naux; ++k) val += rho[i * naux + k] * eri1[k * nao_pair + j];
  
  if(init) vj[i * nao_pair + j] = val;
  else vj[i * nao_pair + j] += val;
}

/* ---------------------------------------------------------------------- */

#if 1

__global__ void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  const int indx = j * nao;
  for(int k=0; k<nao; ++k) buf[indx+k] = tril[ map[indx+k] ];  
}

#else

__global__ void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao*nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  buf[j] = tril[ map[j] ];
}
#endif

/* ---------------------------------------------------------------------- */
__global__ void _pack_eri1(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
 //eri1 is out, buf2 is in, we are packing buf2 of shape naux * nao * nao to eri1 of shape naux * nao_pair 
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  const int indx = j * nao;
  //for(int k=0; k<nao; ++k) buf[indx+k] = tril[ map[indx+k] ];  
  for(int k=0; k<nao; ++k) tril[map[indx+k]] = buf[ indx+k ];  
}

/* ---------------------------------------------------------------------- */


#if 1

//https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
// modified to support nonsquare matrices
__global__ void _transpose(double * out, double * in, int nrow, int ncol)
{
  __shared__ double cache[_TRANSPOSE_BLOCK_SIZE][_TRANSPOSE_BLOCK_SIZE+1];
  
  int irow = blockIdx.x * _TRANSPOSE_BLOCK_SIZE + threadIdx.x;
  int icol = blockIdx.y * _TRANSPOSE_BLOCK_SIZE + threadIdx.y;

  // load tile into fast local memory

  const int indxi = irow * ncol + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < nrow && (icol+i) < ncol) // nonsquare
      cache[threadIdx.y + i][threadIdx.x] = in[indxi + i]; // threads read chunk of a row and write as a column
  }

  // block to ensure reads finish
  
  __syncthreads();

  // swap indices
  
  irow = blockIdx.y * _TRANSPOSE_BLOCK_SIZE + threadIdx.x;
  icol = blockIdx.x * _TRANSPOSE_BLOCK_SIZE + threadIdx.y;

  // write tile to global memory

  const int indxo = irow * nrow + icol;
  for(int i=0; i<_TRANSPOSE_BLOCK_SIZE; i+= _TRANSPOSE_NUM_ROWS) {
    if(irow < ncol && (icol + i) < nrow) // nonsquare
      out[indxo + i] = cache[threadIdx.x][threadIdx.y + i];
  }
}

#else

__global__ void _transpose(double * buf3, double * buf1, int nrow, int ncol)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= nrow) return;

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  while (j < ncol) {
    buf3[j*nrow + i] = buf1[i*ncol + j]; // these writes should be to SLM and then contiguous chunks written to global memory
    j += blockDim.y;
  }
  
}

#endif

__global__ void _get_bufd( const double* bufpp, double* bufd, int naux, int nmo){
    
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < naux && j < nmo) {
        bufd[i * nmo + j] = bufpp[(i*nmo + j)*nmo + j];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _get_bufpa (const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= naux) return;
  if(j >= nmo) return;
  if(k >= ncas) return;
  
  int inputIndex = (i*nmo + j)*nmo + k+ncore;
  int outputIndex = (i*nmo + j)*ncas + k;
  bufpa[outputIndex] = bufpp[inputIndex];
}

/* ---------------------------------------------------------------------- */
__global__ void _get_bufaa (const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= naux) return;
  if(j >= ncas) return;
  if(k >= ncas) return;
  
  int inputIndex = (i*nmo + (j+ncore))*nmo + k+ncore;
  int outputIndex = (i*ncas + j)*ncas + k;
  bufaa[outputIndex] = bufpp[inputIndex];
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_120(double * in, double * out, int naux, int nao, int ncas) {
    //Pum->muP
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= naux) return;
    if(j >= ncas) return;
    if(k >= nao) return;

    int inputIndex = i*nao*ncas+j*nao+k;
    int outputIndex = j*nao*naux  + k*naux + i;
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

__global__ void _get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao) {
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncas && j < nao) {
        small_mat[i * nao + j] = big_mat[j*nao + i+ncore];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_210(double * in, double * out, int naux, int nao, int ncas) {
    //Pum->muP
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= naux) return;
    if(j >= ncas) return;
    if(k >= nao) return;

    int inputIndex = i*nao*ncas+j*nao+k;
    int outputIndex = k*ncas*naux  + j*naux + i;
    out[outputIndex] = in[inputIndex];
}

/* ---------------------------------------------------------------------- */

__global__ void _extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(i >= ncas) return;
  if(j >= ncas) return;
  
  small_mat[i * ncas + j] = big_mat[(i + ncore) * nmo + (j + ncore)];
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_2310(double * in, double * out, int nmo, int ncas) {
    //a.transpose(2,3,1,0)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= nmo) return;
    if(j >= ncas) return;
    if(k >= ncas) return;

    for(int l=0; l<ncas; ++l) {
      int inputIndex = ((i*ncas+j)*ncas+k)*ncas+l;
      int outputIndex = k*ncas*ncas*nmo + l*ncas*nmo + j*nmo + i;
      out[outputIndex] = in[inputIndex];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _transpose_3210(double* in, double* out, int nmo, int ncas) {
    //a.transpose(3,2,1,0)-ncas,ncas,ncas,nmo
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ncas) return;
    if(j >= ncas) return;
    if(k >= ncas) return;
    
    for(int l=0;l<nmo;++l){
      int inputIndex = ((i*ncas+j)*ncas+k)*nmo+l;
      int outputIndex = l*ncas*ncas*ncas+k*ncas*ncas+j*ncas+i;
      out[outputIndex]=in[inputIndex];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void _pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= nmo) return;
  if(j >= ncas) return;
  if(k >= ncas_pair) return;

  double * out_buf = &(out[(i*ncas + j) * ncas_pair]);
  double * in_buf = &(in[(i*ncas + j) * ncas*ncas]);

  out_buf[ k ] = in_buf[ map[k] ];
}

/* ---------------------------------------------------------------------- */

__global__ void _unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nmo*ncas) return;
  if(j >= ncas*ncas) return;

  double * in_buf = &(in[i * ncas_pair ]);
  double * out_buf = &(out[i * ncas*ncas ]);

  out_buf[j] = in_buf[ map[j] ];
}

/* ---------------------------------------------------------------------- */

__global__ void _pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]=in[j*ncas*nmo + i];

}

/* ---------------------------------------------------------------------- */

__global__ void _pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]+=in[j*ncas*nmo + i]; // this doesn't work because map spans (ncas x ncas) and has duplicate entries
}

/* ---------------------------------------------------------------------- */

__global__ void _vecadd(const double * in, double * out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N) return;
    out[i] += in[i];
}
/* ---------------------------------------------------------------------- */
__global__ void _get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= ngrid) return;

    Pi[i] += rho[i] * rho[i];
}  
/* ---------------------------------------------------------------------- */
__global__ void _make_gridkern(double * mo_grid, double * gridkern, int ngrid, int ncas)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ngrid) return;
    if(j >= ncas) return;
    if(k >= ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_mo_grid = &(mo_grid[i*ncas]);
    tmp_gridkern[j*ncas+k] = tmp_mo_grid[j]*tmp_mo_grid[k];
} 
/* ---------------------------------------------------------------------- */
__global__ void _make_buf_pdft(double * gridkern, double * cascm2, double * out, int ngrid, int ncas)
{
    //TODO: convert this to a dgemm
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ngrid) return;
    if(j >= ncas*ncas) return;
    if(k >= ncas*ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_cascm2 = &(cascm2[j*ncas*ncas]);
    double * tmp_out = &(out[i*ncas*ncas+j]);
    tmp_out[0] += tmp_gridkern[k]*tmp_cascm2[k];
} 
/* ---------------------------------------------------------------------- */
__global__ void _make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
    //TODO: convert this to a dgemm
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= ngrid) return;
    if(j >= ncas*ncas) return;
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_buf = &(buf[i*ncas*ncas]);
    double * tmp_Pi = &(Pi[i]);
    tmp_Pi[0] += tmp_gridkern[j]*tmp_buf[j];
} 
/* ---------------------------------------------------------------------- */
__global__ void _set_to_zero(double * array, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=size) return; 
    array[i] = 0.0;
}
/* ---------------------------------------------------------------------- */
__global__ void _compute_FCItrans_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
    int str0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(str0 >= na) return;
    if(j >= nlinka) return;
    #ifdef _ACCELERATE_KERNEL
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
    #else
    int a  = link_index[4*str0*nlinka + 4*j];
    int i  = link_index[4*str0*nlinka + 4*j + 1];
    int str1  = link_index[4*str0*nlinka + 4*j + 2];
    int sign  = link_index[4*str0*nlinka + 4*j + 3];
    double * pket = &(ciket[str0*nb]);
    double * pbra = &(cibra[str1*nb]);
    for (int k=0; k<nb; ++k){
       atomicAdd(&(rdm[a*norb+i]), sign*pbra[k]*pket[k]);
    }
    #endif
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
    #ifdef _ACCELERATE_KERNEL
    int * tab  = &(link_index[4*nlinkb*k+4*j]);
    int a = tab[0];
    int i = tab[1];
    int str1 = tab[2];
    int sign = tab[3];
    #else
    int a  = link_index[4*nlinkb*k+4*j];
    int i  = link_index[4*nlinkb*k+4*j+1];
    int str1  = link_index[4*nlinkb*k+4*j+2];
    int sign  = link_index[4*nlinkb*k+4*j+3];
    #endif
    //rdm[a*norb + i] += sign*pbra[str1]*tmp; //doesn't work when race conditions are present with multiple x,y threads are trying to write to the same combination of a,i in rdm memory block
    atomicAdd(&(rdm[a*norb + i]), sign*pbra[str1]*tmp);
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
    int a = link_index[4*nlinka*stra_id + 4*j]; 
    int i = link_index[4*nlinka*stra_id + 4*j + 1]; 
    int str1 = link_index[4*nlinka*stra_id + 4*j + 2]; 
    int sign = link_index[4*nlinka*stra_id + 4*j + 3];
    
    //double * pci = &(ci[str1*nb]);
    //double * pbuf = &(buf[k*norb2 + i*norb + a]);
    // pbuf[k*norb2] += pci[k]*sign;
    #ifdef _DEBUG_ATOMICADD
    atomicAdd(&(buf[k*norb2 + i*norb + a]), sign*ci[str1*nb + k]);
    #else
    buf[k*norb2 + i*norb + a] += sign*ci[str1*nb + k];
    #endif
    //printf("stra_id: %i str1: %i k: %i a: %i i: %i j: %i sign: %i pdm_location: %i ci_location: %i added: %f , after: %f \n",stra_id, str1,k, a,i,j,sign,k*norb2+i*norb+a, str1*nb+k, ci[str1*nb+k], buf[k*norb2+i*norb+a] );
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
    //printf("stra_id: %i str1: %i str0: %i a: %i i: %i j: %i sign: %i added: %f , prev: %f \n",stra_id, str1,str0, a,i,j,sign, sign*ci[stra_id*nb+str1], buf[str0*norb2+i*norb+a] );
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
__global__ void _compute_FCIrdm3h_a_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= nlinka) return;
    //if (k >= nb) return;//perhaps k can be looped over, and completely avoided if str1 is not in between ia-ja and ib-jb
    int norb2 = norb*norb;
    int * tab = &(link_index[4*nlinka*stra_id + 4*j]); 
    for (int k=ib; k<jb; ++k){//k is the beta loop
      int sign = tab[3];
      if (sign != 0) {
        int str1 = tab[2];
        if ((str1>=ia) && (str1<ja)){//str1 is alpha loop
          int a = tab[0];
          int i = tab[1];
          atomicAdd(&(buf[k*norb2 + i*norb + a]), sign*ci[str1*nb + k]);
          }
        }
      }
}
/* ---------------------------------------------------------------------- */
__global__ void _compute_FCIrdm3h_b_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
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
        atomicAdd(&(buf[str0*norb2 + i*norb + a]), sign*ci[stra_id*nb + str1]);//stra_id is already taken care of in the call itself, maybe work that in the earlier call.
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
__global__ void _veccopy(const double * src, double *dest, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dest[i] = src[i];
} 

/* ---------------------------------------------------------------------- */
__global__ void _gemv_fix(const double * mat, const double * vec, double * out, const int norb2, const int nb, const double alpha, const double beta)
{
    //convert to gemv, shouldn't need this    
    //for (int j=0;j<norb2;++j){for (int i=0;i<nb;++i){ h_tdm1[j] += h_buf1[i*norb2+j]*h_vec[i] ;}}
    //beta is one
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i>=nb) return;
    if (j>=norb2) return;
    //atomicAdd(&(out[j]),mat[i*norb2+j]*vec[i]);
    double buf = 0.0;
    for (int i=0; i<nb; ++i){buf += mat[i*norb2+j]*vec[i];}
    //out[j] += mat[i*norb2+j]*vec[i];
    out[j] += buf;
}
/* ---------------------------------------------------------------------- */
__global__ void _gemm_fix(const double * buf1, const double * buf2, double * out, const int norb2, const int nb)
{
    //convert to gemm, shouldn't need this    
    //i<norb2,k<norb2,j<nb
    //tmp+=h_buf1[j*norb2+i]*h_buf2[j*norb2+k]; 
    //h_pdm2[k*norb2+i]+=tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=norb2) return;
    if (k>=norb2) return;
    double tmp=0.0;
    for (int j=0;j<nb; ++j){
      tmp+=buf1[j*norb2+i]*buf2[j*norb2+k];}
    out[k*norb2+i]+=tmp;
}
/* ---------------------------------------------------------------------- */
void _add_rdm1_to_2(double * dm1, double * dm2, int norb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=norb) return;
    if (j>=norb) return;
    if (k>=norb) return;
    //double * tmp_rdm2 = &(dm2[((i*norb+j)*norb+j)*norb + k]);
    //double * tmp_rdm1 = &(dm1[i*norb + k]);
    dm2[((i*norb+j)*norb+j)*norb + k] -= dm1[i*norb + k];
}
/* ---------------------------------------------------------------------- */
void _add_rdm_transpose(double * buf, double * dm2, int norb)
{
    int norb2 = norb*norb
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=norb2) return;
    if (j>=norb2) return;
    dm2[i*norb2 + j] = (dm2[i*norb2+j] + buf[j*norb2+i])/2;
}

/* ---------------------------------------------------------------------- */

/* Interface functions calling CUDA kernels
/* ---------------------------------------------------------------------- */

void Device::getjk_rho(double * rho, double * dmtril, double * eri, int nset, int naux, int nao_pair)
{
#if 1
  dim3 grid_size(nset, naux, 1);
  dim3 block_size(1, 1, _RHO_BLOCK_SIZE);
#else
  dim3 grid_size(nset, (naux + (_RHO_BLOCK_SIZE - 1)) / _RHO_BLOCK_SIZE, 1);
  dim3 block_size(1, _RHO_BLOCK_SIZE, 1);
#endif

  cudaStream_t s = *(pm->dev_get_queue());
  
  _getjk_rho<<<grid_size, block_size, 0, s>>>(rho, dmtril, eri, nset, naux, nao_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_rho :: nset= %i  naux= %i  RHO_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nset, naux, _RHO_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::getjk_vj(double * vj, double * rho, double * eri, int nset, int nao_pair, int naux, int init)
{
  dim3 grid_size(nset, (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, 1);
  dim3 block_size(1, _DOT_BLOCK_SIZE, 1);
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _getjk_vj<<<grid_size, block_size, 0, s>>>(vj, rho, eri, nset, nao_pair, naux, init);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_vj :: nset= %i  nao_pair= %i _DOT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nset, nao_pair, _DOT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::getjk_unpack_buf2(double * buf2, double * eri, int * map, int naux, int nao, int nao_pair)
{
#if 1
  dim3 grid_size(naux, _TILE(nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#else
  dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _getjk_unpack_buf2<<<grid_size, block_size, 0, s>>>(buf2, eri, map, naux, nao, nao_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_unpack_buf2 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */
void Device::pack_eri(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
#if 1
  //dim3 grid_size(naux, _TILE(nao, _UNPACK_BLOCK_SIZE), 1);
  //dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(naux, nao, 1);
  dim3 block_size(1,1, 1);
#else
  dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif
  cudaStream_t s = *(pm->dev_get_queue());
  
  _pack_eri1<<<grid_size, block_size, 0, s>>>(eri1, buf2, map, naux, nao, nao_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: _pack_eri1 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */


void Device::transpose(double * out, double * in, int nrow, int ncol)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- calling _transpose()\n");
#endif
            
#if 1
  dim3 grid_size( _TILE(nrow, _TRANSPOSE_BLOCK_SIZE), _TILE(ncol, _TRANSPOSE_BLOCK_SIZE), 1);
  dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
#else
  dim3 grid_size(nrow, 1, 1);
  dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
#endif
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _transpose<<<grid_size, block_size, 0, s>>>(out, in, nrow, ncol);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- transpose :: nrow= %i  ncol= %i _TRANSPOSE_BLOCK_SIZE= %i  _TRANSPOSE_NUM_ROWS= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nrow, ncol, _TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

__global__ void pack_Mwuv(double *in, double *out, int * map,int nao, int ncas,int ncas_pair)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=nao) return;
    if (j>=ncas) return;
    if (k>=ncas*ncas) return;
    int inputIndex = i*ncas*ncas*ncas+j*ncas*ncas+k;
    int outputIndex = j*ncas_pair*nao+i*ncas_pair+map[k];
    out[outputIndex]=in[inputIndex];
} 

/* ---------------------------------------------------------------------- */

void Device::get_bufpa(const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE,_UNPACK_BLOCK_SIZE,1);
  dim3 grid_size (_TILE(naux, block_size.x), _TILE(nmo, block_size.y), ncas);
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _get_bufpa<<<grid_size, block_size, 0, s>>>(bufpp, bufpa, naux, nmo, ncore, ncas);
}
/* ---------------------------------------------------------------------- */

void Device::get_bufaa(const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE,1,1);
  dim3 grid_size (_TILE(naux, block_size.x), ncas, ncas);
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _get_bufaa<<<grid_size, block_size, 0, s>>>(bufpp, bufaa, naux, nmo, ncore, ncas);
}


/* ---------------------------------------------------------------------- */

void Device::transpose_120(double * in, double * out, int naux, int nao, int ncas, int order)
{
  cudaStream_t s = *(pm->dev_get_queue());

  int na = nao;
  int nb = ncas;
  
  if(order == 1) {
    na = ncas;
    nb = nao;
  }
  
  dim3 block_size (1, 1,1);
  dim3 grid_size (_TILE(naux, block_size.x), na, nb); // originally nmo, nmo
  
  _transpose_120<<<grid_size, block_size, 0, s>>>(in, out, naux, nao, ncas);
}

/* ---------------------------------------------------------------------- */

void Device::get_bufd( const double* bufpp, double* bufd, int naux, int nmo)
{
  dim3 block_size (_UNPACK_BLOCK_SIZE,_UNPACK_BLOCK_SIZE,1);
  dim3 grid_size (_TILE(naux, block_size.x),_TILE(nmo, block_size.y),1);
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _get_bufd<<<grid_size, block_size, 0, s>>>(bufpp, bufd, naux, nmo);
}

/* ---------------------------------------------------------------------- */

void Device::transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, 1, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(_TILE(naux,block_size.x), _TILE(ncas,block_size.y), _TILE(nao,block_size.z));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _transpose_210<<<grid_size,block_size, 0, s>>>(in, out, naux, nao, ncas);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- h2eff_df_contract1::transpose_210 :: naux= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 naux, ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(_TILE(ncas,block_size.x), _TILE(ncas,block_size.y));
    
  cudaStream_t s = *(pm->dev_get_queue());
  
  _extract_submatrix<<<grid_size, block_size, 0, s>>>(big_mat, small_mat, ncas, ncore, nmo);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- extract_submatrix :: ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(nmo*ncas,_UNPACK_BLOCK_SIZE), _TILE(ncas*ncas,_UNPACK_BLOCK_SIZE), 1);

  cudaStream_t s = *(pm->dev_get_queue());
  
  _unpack_h2eff_2d<<<grid_size, block_size, 0>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- _unpack_h2eff_2d :: nmo*ncas= %i  ncas*ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo*ncas, ncas*ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::transpose_2310(double * in, double * out, int nmo, int ncas)
{
  dim3 block_size(1,1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(nmo,block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _transpose_2310<<<grid_size, block_size, 0, s>>>(in, out, nmo, ncas);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::transpose_2310 :: nmo= %i  ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo, ncas, _DEFAULT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::transpose_3210(double* in, double* out, int nmo, int ncas)
{
  dim3 block_size(1,1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(ncas,block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _transpose_3210<<<grid_size, block_size, 0, s>>>(in, out, nmo, ncas);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::transpose_3210 :: ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, _DEFAULT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(1, 1, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(nmo, ncas, _TILE(ncas_pair, _DEFAULT_BLOCK_SIZE));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _pack_h2eff_2d<<<grid_size, block_size, 0, s>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::_pack_h2eff_2d :: nmo= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo, ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao)
{
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ncas, block_size.x), _TILE(nao, block_size.y));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _get_mo_cas<<<grid_size, block_size, 0, s>>>(big_mat, small_mat, ncas, ncore, nao);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::_get_mo_cas :: ncas= %i  nao= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, nao, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(nmo*ncas,block_size.x), _TILE(ncas*ncas,block_size.y));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _pack_d_vuwM<<<grid_size,block_size, 0, s>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo*ncas, ncas*ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(nmo*ncas,block_size.x), _TILE(ncas*ncas,block_size.y));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _pack_d_vuwM_add<<<grid_size,block_size, 0, s>>>(in, out, map, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM_add :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nmo*ncas, ncas*ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::vecadd(const double * in, double * out, int N)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _vecadd<<<grid_size,block_size, 0, s>>>(in, out, N);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::vecadd :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 N, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */
void Device::get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(ngrid, block_size.x),1,1);

  cudaStream_t s = *(pm->dev_get_queue());

  _get_rho_to_Pi<<<grid_size, block_size,0, s>>>(rho, Pi, ngrid);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rho_to_Pi :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ngrid, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}
/* ---------------------------------------------------------------------- */
void Device::make_gridkern(double * d_mo_grid, double * d_gridkern, int ngrid, int ncas)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(ngrid, block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));

  cudaStream_t s = *(pm->dev_get_queue());

  _make_gridkern<<<grid_size, block_size,0,s>>>(d_mo_grid, d_gridkern, ngrid, ncas);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_gridkern :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}
/* ---------------------------------------------------------------------- */
void Device::make_buf_pdft(double * gridkern, double * buf, double * cascm2, int ngrid, int ncas)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(ngrid, block_size.x),_TILE(ncas*ncas,block_size.y),_TILE(ncas*ncas,block_size.z));

  cudaStream_t s = *(pm->dev_get_queue());

  // buf = aij, klij ->akl, gridkern, cascm2
  _make_buf_pdft<<<grid_size, block_size,0,s>>>(gridkern, cascm2, buf, ngrid, ncas);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_gridkern :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif

}
/* ---------------------------------------------------------------------- */
void Device::make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(ngrid, block_size.x),_TILE(ncas*ncas,block_size.y),1);

  cudaStream_t s = *(pm->dev_get_queue());

  _make_Pi_final<<<grid_size, block_size,0,s>>>(gridkern, buf, Pi, ngrid, ncas);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_Pi_final; :: Ngrid= %i Ncas =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ngrid, ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCItrans_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nlinka,block_size.y),1);

  cudaStream_t s = *(pm->dev_get_queue());

  _compute_FCItrans_rdm1a<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinka, link_index);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rdm_from_ci; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCItrans_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nb,block_size.y),_TILE(nlinkb, block_size.z));
  
  cudaStream_t s = *(pm->dev_get_queue());


  _compute_FCItrans_rdm1b<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinkb, link_index);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rdm_from_ci; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCImake_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  {dim3 block_size(_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE,1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nlinka, block_size.y),1);
  _compute_FCImake_rdm1a<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinka, link_index);
  #ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::get_rdm_from_ci; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
  #endif
  }
  {dim3 block_size (1,1,1);
   dim3 grid_size(1,1,1);
   _symmetrize_rdm<<<grid_size, block_size, 0, s>>> (norb, rdm);}
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCImake_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  {
  //dim3 block_size(_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE); //TODO: fix this?
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(na, block_size.x),_TILE(nb, block_size.y),_TILE(nlinkb, block_size.z));
  #ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::make_rdm1b; :: Na= %i Nb =%i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 na, nb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  #endif
  _compute_FCImake_rdm1b<<<grid_size, block_size,0,s>>>(cibra, ciket, rdm, norb, na, nb, nlinkb, link_index);
  _CUDA_CHECK_ERRORS();
  }
  {dim3 block_size(1,1,1);
   dim3 grid_size(1,1,1); 
   _symmetrize_rdm<<<grid_size, block_size, 0, s>>> (norb, rdm); }
}

/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm2_a_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int * link_index)
{
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(nlinka, block_size.x), _TILE(nb, block_size.y), 1);
  cudaStream_t s = *(pm->dev_get_queue());
  _compute_FCIrdm2_a_t1ci<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinka, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}  
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm2_b_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinkb, int * link_index)
{
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(nb, block_size.x), _TILE(nlinkb, block_size.y), 1);
  cudaStream_t s = *(pm->dev_get_queue());
  _compute_FCIrdm2_b_t1ci<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinkb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_b_t1ci; :: Nb= %i Norb =%i Nlinkb =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinkb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
} 
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm3h_a_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  dim3 block_size(1,1,1);
  //dim3 grid_size(_TILE(nlinka, block_size.x), _TILE(nb, block_size.y), 1);
  dim3 grid_size(_TILE(nlinka, block_size.x), 1, 1);
  cudaStream_t s = *(pm->dev_get_queue());
  _compute_FCIrdm3h_a_t1ci<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinka, ia, ja, ib, jb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}  
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm3h_b_t1ci(double * ci, double * buf, int stra_id, int nb, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  if ((stra_id>=ia) && stra_id<ja){ //I'm writing this in, but buf being zero needs to be accounted in the full function call as well
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(nb, block_size.x), _TILE(nlinkb, block_size.y), 1);
  cudaStream_t s = *(pm->dev_get_queue());
  _compute_FCIrdm3h_b_t1ci<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinkb, ia, ja, ib, jb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_b_t1ci; :: Nb= %i Norb =%i Nlinkb =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinkb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
  }
} 
/* ---------------------------------------------------------------------- */
void Device::gemv_fix(const double * buf, const double * bravec, double * pdm1, const int norb2, const int nb, const double alpha, const double beta)
{
  //dim3 block_size(_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE,1);
  //dim3 grid_size(_TILE(nb, block_size.x), _TILE(norb2,block_size.y), 1);
  dim3 block_size(_DEFAULT_BLOCK_SIZE,1,1);
  dim3 grid_size(_TILE(norb2, block_size.x),1, 1);
  cudaStream_t s = *(pm->dev_get_queue());
  _gemv_fix<<<grid_size, block_size, 0, s>>>(buf, bravec, pdm1, norb2, nb, alpha, beta);
  _CUDA_CHECK_ERRORS();
}
/* ---------------------------------------------------------------------- */
void Device::gemm_fix(const double * buf1, const double * buf2, double * pdm2, const int norb2, const int nb)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE,_DEFAULT_BLOCK_SIZE,1);
  dim3 grid_size(_TILE(norb2, block_size.x), _TILE(norb2,block_size.y), 1);
  cudaStream_t s = *(pm->dev_get_queue());
  _gemm_fix<<<grid_size, block_size, 0, s>>>(buf1, buf2, pdm2, norb2, nb);
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */
void Device::transpose_jikl(double * tdm, double * buf, int norb)
{
  int norb2 = norb*norb;
  cudaStream_t s = *(pm->dev_get_queue());
  {
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1,1); 
  dim3 grid_size(_TILE(norb2, block_size.x), 1,1);
  _transpose_jikl<<<grid_size,block_size,0,s>>>(tdm,buf,norb);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::transpose_jikl; :: Norb= %i grid_size= %i %i %i  block_size= %i %i %i\n",
	 norb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif  
  }
  {
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(norb2*norb2, block_size.x), 1, 1);
  _veccopy<<<grid_size, block_size, 0,s>>>(buf, tdm, norb2*norb2); 
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::copy_tdm; :: Norb= %i grid_size= %i %i %i  block_size= %i %i %i\n",
	 norb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif  
  }
}
/* ---------------------------------------------------------------------- */
void Device::reorder(double * dm1, double * dm2, double * buf, int norb)
{
  int norb2 = norb*norb;
  cudaStream_t s = *(pm->dev_get_queue());
  //for k in range (norb): rdm2[:,k,k,:] -= rdm1.T //remember, rdm1 is returned as rdm1.T, so double transpose, hence just rdm1
  //rdm2 = (rdm2+rdm2.transpose(2,3,0,1))/2
  {
    dim3 block_size (1,1,1);
    dim3 grid_size (_TILE(norb, block_size.x), _TILE(norb, block_size.y), _TILE(norb, block_size.z));
    _add_rdm1_to_2 <<<grid_size, block_size, 0, s>>> (dm1, dm2, norb);
    _CUDA_CHECK_ERRORS();
  }
  {
    dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
    dim3 grid_size(_TILE(norb2*norb2, block_size.x), 1, 1);
    _veccopy<<<grid_size, block_size, 0,s>>>(rdm2, buf, norb2*norb2); 
    _CUDA_CHECK_ERRORS();
  }
  { 
    dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
    dim3 grid_size (_TILE(norb2, block_size.x), _TILE(norb2, block_size.y),1);
    _add_rdm_transpose(buf, rdm2, norb) 
  }
  
    
   
}
/* ---------------------------------------------------------------------- */
void Device::set_to_zero(double * array, int size)
{
  cudaStream_t s = *(pm->dev_get_queue());
  #if 1
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(size, block_size.x),1,1);
  _set_to_zero<<<grid_size, block_size, 0,s>>>(array, size);
  _CUDA_CHECK_ERRORS();
 #else
 cudaMemSet(array,0, size*sizeof(double), s); //Is this better?
 #endif
}



#endif
