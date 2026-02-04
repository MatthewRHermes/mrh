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

__global__ void _vecadd_batch(const double * in, double * out, int N, int num_batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N) return;

    double val = 0.0;
    for(int j=0; j<num_batches; ++j) val += in[j*N + i];
    
    out[i] += val;
}

/* ---------------------------------------------------------------------- */

__global__ void _memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N) return;
    
    for(int j=0; j<num_batches; ++j) inout[j*stride + offset + i] = 0.0;
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
__global__ void _veccopy(const double * src, double *dest, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dest[i] = src[i];
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
__global__ void _transpose_021(double * in, double * out, int ax1, int ax2, int ax3) {
    // abc->acb
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ax1) return;
    if(j >= ax2) return;
    if(k >= ax3) return;

    int inputIndex = (i*ax3+k)*ax2+j;
    int outputIndex = (i*ax2+j)*ax3+k;
    //printf("%i %i %i %f\n",i, j, k, in[inputIndex]);
    out[outputIndex] = in[inputIndex];
}
/* ---------------------------------------------------------------------- */
__global__ void _transpose_102(double * in, double * out, int ax1, int ax2, int ax3) {
    // abc->acb
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= ax1) return;
    if(j >= ax2) return;
    if(k >= ax3) return;

    int inputIndex = (j*ax1+i)*ax3+k;
    int outputIndex = (i*ax2+j)*ax3+k;
    //printf("%i %i %i %f\n",i, j, k, in[inputIndex]);
    out[outputIndex] = in[inputIndex];
}


/* ---------------------------------------------------------------------- */
__global__ void _transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4) {
    // rs(bazl)k->(bazl)skr
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;

    if(idx1 >= ax1) return;
    if(idx2 >= ax2) return;
    if(idx3 >= ax3) return;
    int inputIndex, outputIndex;
    for (int idx4=0;idx4<ax4;++idx4){
      outputIndex = ((idx3*ax2 + idx2)*ax4 + idx4)*ax1 + idx1; 
      inputIndex = ((idx1*ax2 + idx2)*ax3 + idx3)*ax4 + idx4;
      //printf("input: %i: %f output: %i\n",inputIndex, in[inputIndex], outputIndex);
      out[outputIndex] = in[inputIndex];}
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

void Device::vecadd_batch(const double * in, double * out, int N, int num_batches)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _vecadd_batch<<<grid_size, block_size, 0, s>>>(in, out, N, num_batches);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::vecadd_batch :: N= %i  num_batches= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 N, num_batches, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _CUDA_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void Device::memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  cudaStream_t s = *(pm->dev_get_queue());
  
  _memset_zero_batch_stride<<<grid_size, block_size, 0, s>>>(inout, stride, offset, N, num_batches);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::memset_zero_batch_stride :: stride= %i  offset= %i  N= %i  num_batches= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 stride, offset, N, num_batches, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
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
void Device::compute_FCItrans_rdm1a_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinka, 
                                        int ia_bra, int ja_bra, int ib_bra, int jb_bra, 
                                        int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sign, 
                                        int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
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
    //printf("na_ket: %i ia_ket: %i ja_ket: %i ib_ket: %i ib_bra: %i nb_bra: %i nb_ket: %i b_len: %i b_bra_offset: %i b_ket_offset: %i sign: %i\n",na_ket, ia_ket, ja_ket, ib_ket, ib_bra, nb_bra, nb_ket, b_len, b_bra_offset, b_ket_offset, sign);
#endif
  _CUDA_CHECK_ERRORS();
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCItrans_rdm1b_v2( double * cibra, double * ciket, double * rdm, int norb, int nlinkb, 
                                        int ia_bra, int ja_bra, int ib_bra, int jb_bra, 
                                        int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sign, 
                                        int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());

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
  _CUDA_CHECK_ERRORS();
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
  #endif
  _CUDA_CHECK_ERRORS();
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
void Device::compute_FCIrdm2_a_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinka, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(1, 1, _DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches, block_size.x),_TILE(nb, block_size.y), 1);
  _compute_FCIrdm2_a_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id,batches,nb, norb, nlinka, link_index);
  _CUDA_CHECK_ERRORS();
  printf("compute_FCIrdm2_a_t1ci_v2 working\n");
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
  _CUDA_CHECK_ERRORS();
}  
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm2_b_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinkb, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  {dim3 block_size(1, 1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches,block_size.x),_TILE(nb, block_size.y), 1);
  _compute_FCIrdm2_b_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, batches, nb, norb, nlinkb, link_index);}
  _CUDA_CHECK_ERRORS();
#ifdef _DEBUG_DEVICE 
#endif
} 

/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm3h_a_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  #if 0
  dim3 block_size(_DEFAULT_BLOCK_SIZE,1,1);
  dim3 grid_size(_TILE(jb-ib, block_size.x), 1, 1);
  _compute_FCIrdm3h_a_t1ci_v3<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinka, ia, ja, ib, jb, link_index);
  #else
  dim3 block_size(1,_DEFAULT_BLOCK_SIZE,1);
  dim3 grid_size(_TILE(jb-ib, block_size.x), 1, 1);
  _compute_FCIrdm3h_a_t1ci_v4<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, nb, norb, nlinka, ia, ja, ib, jb, link_index);
  #endif
  _CUDA_CHECK_ERRORS();
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm3h_b_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
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
  _CUDA_CHECK_ERRORS();
#endif
}  
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm3h_a_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_ci, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(1, 1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches,block_size.x), _TILE(jb-ib, block_size.y), 1);
  _compute_FCIrdm3h_a_t1ci_v5<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, batches, nb, nb_ci, norb, nlinka, ia, ja, ib, jb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_a_t1ci; :: Nb= %i Norb =%i Nlinka =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinka, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
  _CUDA_CHECK_ERRORS();
}
/* ---------------------------------------------------------------------- */
void Device::compute_FCIrdm3h_b_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(1, 1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(batches, block_size.x), _TILE(nb, block_size.y), 1);
  _compute_FCIrdm3h_b_t1ci_v5<<<grid_size, block_size, 0,s>>>(ci, buf, stra_id, batches, nb, nb_bra, norb, nlinkb, ia, ja, ib, jb, link_index);
#ifdef _DEBUG_DEVICE 
  printf("LIBGPU ::  -- general::compute_FCIrdm2_b_t1ci; :: Nb= %i Norb =%i Nlinkb =%i grid_size= %i %i %i  block_size= %i %i %i\n",
	 nb, norb, nlinkb, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
#endif
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
void Device::reduce_buf3_to_rdm(const double * buf3, double * dm2, int size_tdm2, int num_gemm_batches)
{
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1,1);
  dim3 grid_size (_TILE(size_tdm2, block_size.x),1,1);
  _vecadd_batch<<<grid_size, block_size,0, s>>>(buf3, dm2, size_tdm2, num_gemm_batches);
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */
void Device::reorder(double * dm1, double * dm2, double * buf, int norb)
{
  cudaStream_t s = *(pm->dev_get_queue());
  //for k in range (norb): rdm2[:,k,k,:] -= rdm1.T //remember, rdm1 is returned as rdm1.T, so double transpose, hence just rdm1
  {
    dim3 block_size (1,1,1);
    dim3 grid_size (_TILE(norb, block_size.x), _TILE(norb, block_size.y), _TILE(norb, block_size.z));
    _add_rdm1_to_2<<<grid_size, block_size, 0, s>>> (dm1, dm2, norb);
    _CUDA_CHECK_ERRORS();
  }
  //rdm2 = (rdm2+rdm2.transpose(2,3,0,1))/2
  #if 0
  //this is for reducing numerical error ... we can implement it later
  {
    dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
    dim3 grid_size(_TILE(norb2*norb2, block_size.x), 1, 1);
    _veccopy<<<grid_size, block_size, 0,s>>>(dm2, buf, norb2*norb2); 
    _CUDA_CHECK_ERRORS();
  }
  { 
    dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
    dim3 grid_size (_TILE(norb2, block_size.x), _TILE(norb2, block_size.y),1);
    _add_rdm_transpose<<<grid_size, block_size, 0, s>>>(buf, dm2, norb); 
    _CUDA_CHECK_ERRORS();
  }
  {
    dim3 block_size(_DEFAULT_BLOCK_SIZE, 1,1); 
    dim3 grid_size(_TILE(norb2*norb2, block_size.x), 1,1);
    _build_rdm<<<grid_size, block_size, 0>>>(buf, dm2, norb2*norb2);
    _CUDA_CHECK_ERRORS();
  }
  #endif
  //axpy pending from buf2 to rdm2 
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
/* ---------------------------------------------------------------------- */
void Device::filter_sfudm( const double * dm2, double * dm1, int norb)
{
  //only need dm2[-1,:-1, :-1, -1]
  cudaStream_t s = *(pm->dev_get_queue());
  int norb_m1 = norb-1;
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(norb_m1, block_size.x),_TILE(norb_m1, block_size.y),1);
  _filter_sfudm<<<grid_size, block_size, 0,s>>>(dm2,dm1,norb_m1);
  _CUDA_CHECK_ERRORS();
}
/* ---------------------------------------------------------------------- */
void Device::filter_tdmpp( const double * dm2, double * dm1, int norb, int spin)
{
  //only need dm2[:-ndum,-1,:-ndum,-ndum] //ndum = 2-(spin%2)
  int ndum = (spin!=1) ? 2:1; 
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(norb-ndum, block_size.x),_TILE(norb-ndum, block_size.y),1);
  _filter_tdmpp<<<grid_size, block_size, 0,s>>>(dm2,dm1,norb,spin);
  _CUDA_CHECK_ERRORS();
}
/* ---------------------------------------------------------------------- */
void Device::filter_tdm1h( const double * in, double * out, int norb)
{
  //tdm1h = tdm1h.T
  //tdm1h = tdm1h[-1,:-1]
  //in is (norb+1)^2
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(norb, block_size.x),1,1);
  _filter_tdm1h<<<grid_size, block_size, 0,s>>>(in,out,norb);
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */
void Device::filter_tdm3h(double * in, double * out, int norb)
{
  //tdm3h = tdm3h[:-1,-1,:-1,:-1]
  //dm2 is (norb+1)^4
  cudaStream_t s = *(pm->dev_get_queue());
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(norb, block_size.x),_TILE(norb, block_size.y),_TILE(norb, block_size.z));
  _filter_tdm3h<<<grid_size, block_size, 0,s>>>(in, out,norb);
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void Device::veccopy(const double * src, double *dest, int size)
{
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(size, block_size.x), 1, 1);
  _veccopy<<<grid_size, block_size, 0, s>>>(src, dest, size);
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void Device::transpose_021( double * in, double * out, int ax1, int ax2, int ax3)
{
  cudaStream_t s = *(pm->dev_get_queue());
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ax1, block_size.x),_TILE(ax2, block_size.y),_TILE(ax3, block_size.z));
  #if 1
  _transpose_021<<<grid_size, block_size, 0, s>>>(in, out, ax1, ax2, ax3);
  #else

  #endif
  _CUDA_CHECK_ERRORS();
}
/* ---------------------------------------------------------------------- */
void Device::transpose_102( double * in, double * out, int ax1, int ax2, int ax3)
{
  cudaStream_t s = *(pm->dev_get_queue());
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ax1, block_size.x),_TILE(ax2, block_size.y),_TILE(ax3, block_size.z));
  #if 1
  _transpose_102<<<grid_size, block_size, 0, s>>>(in, out, ax1, ax2, ax3);
  #else

  #endif
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */
void Device::transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4)
{
  cudaStream_t s = *(pm->dev_get_queue());
  dim3 block_size(1, 1,1);
  dim3 grid_size(_TILE(ax1, block_size.x),_TILE(ax2, block_size.y),_TILE(ax3,block_size.z));
  _transpose_2130<<<grid_size, block_size, 0, s>>>(in, out, ax1, ax2, ax3, ax4);
  _CUDA_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */




#endif
