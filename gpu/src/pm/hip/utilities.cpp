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

/* ---------------------------------------------------------------------- */

#if 1

//https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/hip-cpp/transpose/transpose.cu

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

__global__ void _set_to_zero(double * array, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=size) return; 
    array[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

__global__ void _veccopy(const double * src, double *dest, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dest[i] = src[i];
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
    // abc->bac
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

void DeviceUtils::transpose(double * out, double * in, int nrow, int ncol)
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
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _transpose<<<grid_size, block_size, 0, s>>>(out, in, nrow, ncol);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- transpose :: nrow= %i  ncol= %i _TRANSPOSE_BLOCK_SIZE= %i  _TRANSPOSE_NUM_ROWS= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 nrow, ncol, _TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::vecadd(const double * in, double * out, int N)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _vecadd<<<grid_size,block_size, 0, s>>>(in, out, N);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::vecadd :: N= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 N, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::vecadd_batch(const double * in, double * out, int N, int num_batches)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _vecadd_batch<<<grid_size, block_size, 0, s>>>(in, out, N, num_batches);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::vecadd_batch :: N= %i  num_batches= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 N, num_batches, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _memset_zero_batch_stride<<<grid_size, block_size, 0, s>>>(inout, stride, offset, N, num_batches);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- general::memset_zero_batch_stride :: stride= %i  offset= %i  N= %i  num_batches= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 stride, offset, N, num_batches, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::set_to_zero(double * array, int size)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  #if 1
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(size, block_size.x),1,1);
  _set_to_zero<<<grid_size, block_size, 0,s>>>(array, size);
  _HIP_CHECK_ERRORS();
 #else
 cudaMemSet(array,0, size*sizeof(double), s); //Is this better?
 #endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::veccopy(const double * src, double *dest, int size)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(size, block_size.x), 1, 1);
  _veccopy<<<grid_size, block_size, 0, s>>>(src, dest, size);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_021( double * in, double * out, int ax1, int ax2, int ax3)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ax1, block_size.x),_TILE(ax2, block_size.y),_TILE(ax3, block_size.z));

  _transpose_021<<<grid_size, block_size, 0, s>>>(in, out, ax1, ax2, ax3);

  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_102( double * in, double * out, int ax1, int ax2, int ax3)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  //dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE);
  dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ax1, block_size.x),_TILE(ax2, block_size.y),_TILE(ax3, block_size.z));
  
  _transpose_102<<<grid_size, block_size, 0, s>>>(in, out, ax1, ax2, ax3);

  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(1, 1,1);
  dim3 grid_size(_TILE(ax1, block_size.x),_TILE(ax2, block_size.y),_TILE(ax3,block_size.z));
  _transpose_2130<<<grid_size, block_size, 0, s>>>(in, out, ax1, ax2, ax3, ax4);
  _HIP_CHECK_ERRORS();
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_3210(double* in, double* out, int nmo, int ncas)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());
  dim3 block_size(1,1,_DEFAULT_BLOCK_SIZE);
  dim3 grid_size(_TILE(ncas,block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));
  
  _transpose_3210<<<grid_size, block_size, 0, s>>>(in, out, nmo, ncas);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- update_h2eff_sub::transpose_3210 :: ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 ncas, _DEFAULT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceUtils::transpose_120(double * in, double * out, int naux, int nao, int ncas, int order)
{
  hipStream_t s = *(ctx.pm->dev_get_queue());

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

void DeviceUtils::getjk_unpack_buf2(double * buf2, double * eri, int * map, int naux, int nao, int nao_pair)
{
#if 1
  dim3 grid_size(naux, _TILE(nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#else
  dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif
  
  hipStream_t s = *(ctx.pm->dev_get_queue());
  
  _getjk_unpack_buf2<<<grid_size, block_size, 0, s>>>(buf2, eri, map, naux, nao, nao_pair);
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- get_jk::_getjk_unpack_buf2 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	 naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
  _HIP_CHECK_ERRORS();
#endif
}


#endif
