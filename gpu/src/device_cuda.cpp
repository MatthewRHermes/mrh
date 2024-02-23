/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 32
#define _DOT_BLOCK_SIZE 32
#define _TRANSPOSE_BLOCK_SIZE 32
#define _UNPACK_BLOCK_SIZE 32

#define _DEFAULT_BLOCK_SIZE 32

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int _nset, int _nao, int _naux, int count)
{
  //  printf("Inside init_get_jk()\n");
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  blksize = _blksize;
  nset = _nset;
  nao = _nao;
  naux = _naux;

  nao_pair = nao * (nao+1) / 2;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)

  //  double * eri1 = static_cast<double*>(info_eri1.ptr);
  //  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int _size_vj = nset * nao_pair;
  if(_size_vj > size_vj) {
    size_vj = _size_vj;
    //if(vj) pm->dev_free_host(vj);
    //vj = (double *) pm->dev_malloc_host(size_vj * sizeof(double));
    if(d_vj) pm->dev_free(d_vj);
    d_vj = (double *) pm->dev_malloc(size_vj * sizeof(double));
  }
  //for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;
  
  int _size_vk = nset * nao * nao;
  if(_size_vk > size_vk) {
    size_vk = _size_vk;
    //    if(_vktmp) pm->dev_free_host(_vktmp);
    //    _vktmp = (double *) pm->dev_malloc_host(size_vk*sizeof(double));

#ifdef _CUDA_NVTX
    nvtxRangePushA("Realloc");
#endif
    
    if(d_vkk) pm->dev_free(d_vkk);
    d_vkk = (double *) pm->dev_malloc(size_vk * sizeof(double));

#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
  }
  //  for(int i=0; i<_size_vk; ++i) _vktmp[i] = 0.0;

  int _size_buf = blksize * nao * nao;
  if(_size_buf > size_buf) {
    size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double*) pm->dev_malloc_host(2*size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (blksize*nao, nao)

#ifdef _CUDA_NVTX
    nvtxRangePushA("Realloc");
#endif

    if(d_buf1) pm->dev_free(d_buf1);
    if(d_buf2) pm->dev_free(d_buf2);
    if(d_buf3) pm->dev_free(d_buf3);
    
    d_buf1 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf2 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf3 = (double *) pm->dev_malloc(size_buf * sizeof(double));

#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
  }

  int _size_fdrv = nao * nao * num_threads;
  if(_size_fdrv > size_fdrv) {
    size_fdrv = _size_fdrv;
    if(buf_fdrv) pm->dev_free_host(buf_fdrv);
    buf_fdrv = (double *) pm->dev_malloc_host(size_fdrv*sizeof(double));
  }

  int _size_dms = nao * nao;
  if(_size_dms > size_dms) {
    size_dms = _size_dms;
    if(d_dms) pm->dev_free(d_dms);
    d_dms = (double *) pm->dev_malloc(size_dms * sizeof(double));
  }

  int _size_dmtril = nset * nao_pair;
  if(_size_dmtril > size_dmtril) {
    size_dmtril = _size_dmtril;
    if(d_dmtril) pm->dev_free(d_dmtril);
    d_dmtril = (double *) pm->dev_malloc(size_dmtril * sizeof(double));
  }
  
  int _size_eri1 = naux * nao_pair;
  if(_size_eri1 > size_eri1) {
    size_eri1 = _size_eri1;
    if(d_eri1) pm->dev_free(d_eri1);
    d_eri1 = (double *) pm->dev_malloc(size_eri1 * sizeof(double));
  }
  
  int _size_tril_map = nao * nao;
  //  if(_size_tril_map > size_tril_map) {
  if(_size_tril_map != size_tril_map) {
    // nao can change between calls, so mapping needs to be updated...
    // I think there are only two mappings needed
    
    size_tril_map = _size_tril_map;
    if(tril_map) pm->dev_free_host(tril_map);
    if(d_tril_map) pm->dev_free(d_tril_map);
    tril_map = (int *) pm->dev_malloc_host(size_tril_map * sizeof(int));
    d_tril_map = (int *) pm->dev_malloc(size_tril_map * sizeof(int));

    // optimize map later...

    int _i, _j, _ij;
    for(_ij = 0, _i = 0; _i < nao; _i++)
      for(_j = 0; _j<=_i; _j++, _ij++) {
    	tril_map[_i*nao + _j] = _ij;
    	tril_map[_i + nao*_j] = _ij;
      }
    
    pm->dev_push_async(d_tril_map, tril_map, size_tril_map*sizeof(int), stream);
  }
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array_jk[0] += t1 - t0;
#endif
  
  // Create cuda stream
  
  if(stream == nullptr) {
    pm->dev_stream_create(stream);
  }
  
  // Create blas handle

  if(handle == nullptr) {
#ifdef _CUDA_NVTX
    nvtxRangePushA("Create handle");
#endif
    cublasCreate(&handle);
    _CUDA_CHECK_ERRORS();
    cublasSetStream(handle, stream);
    _CUDA_CHECK_ERRORS();
#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
    
#ifdef _SIMPLE_TIMER
  double t2 = omp_get_wtime();
  t_array_jk[1] += t2 - t1;
#endif
  }
  
  //  printf("Leaving init_get_jk()\n");
}

/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk)
{
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  double * vj = static_cast<double*>(info_vj.ptr);
  double * vk = static_cast<double*>(info_vk.ptr);
 
  pm->dev_pull(d_vkk, vk, nset * nao * nao * sizeof(double));
  pm->dev_pull(d_vj, vj, nset * nao_pair * sizeof(double));
}

/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */

__global__ void _getjk_vj(double * vj, double * rho, double * eri1, int nset, int nao_pair, int naux, int count)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nset) return;
  if(j >= nao_pair) return;

  double val = 0.0;
  for(int k=0; k<naux; ++k) val += rho[i * naux + k] * eri1[k * nao_pair + j];
  
  if(count == 0) vj[i * nao_pair + j] = val;
  else vj[i * nao_pair + j] += val;
}

/* ---------------------------------------------------------------------- */

__global__ void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao*nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  buf[j] = tril[ map[j] ];
  //  for(int j=0; j<nao*nao; ++j) buf[j] = tril[ map[j] ];
}

/* ---------------------------------------------------------------------- */

__global__ void _getjk_transpose_buf1_buf3(double * buf3, double * buf1, int naux, int nao)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao) return;

#if 1

  for(int k=0; k<nao; ++k) buf3[k * (naux*nao) + (i*nao+j)] = buf1[(i * nao + j) * nao + k];
  
#else
  DevArray3D da_buf1 = DevArray3D(buf1, naux, nao, nao);
  DevArray2D da_buf3 = DevArray2D(buf3, nao, naux * nao); // python swapped 1st two dimensions?
  
  //  for(int i=0; i<naux; ++i) {
  for(int j=0; j<nao; ++j)
    for(int k=0; k<nao; ++k) da_buf3(k,i*nao+j) = da_buf1(i,j,k);
    //  }
#endif
}

/* ---------------------------------------------------------------------- */

// The _vj and _vk arguements aren't actually used anymore and could be removed. 
void Device::get_jk(int naux,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int count, size_t addr_dfobj)
{
  //  printf("Inside get_jk()\n");
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

#ifdef _CUDA_NVTX
    nvtxRangePushA("GetJK::Init");
#endif
  
  const int with_j = true;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, nao_pair)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int nao_pair = nao * (nao+1) / 2;

  double * d_eri;
#ifndef _USE_ERI_CACHE
   // if not caching, then eri block always transferred
  
  pm->dev_push_async(d_eri1, eri1, naux * nao_pair * sizeof(double), stream);
  d_eri = d_eri1;
#endif
  if(count == 0) pm->dev_push_async(d_dmtril, dmtril, nset * nao_pair * sizeof(double), stream);
  
  int _size_rho = nset * naux;
  if(_size_rho > size_rho) {
    size_rho = _size_rho;
    //    if(rho) pm->dev_free_host(rho);
    if(d_rho) pm->dev_free(d_rho);
    //    rho = (double *) pm->dev_malloc_host(size_rho * sizeof(double));
    d_rho = (double *) pm->dev_malloc(size_rho * sizeof(double));
  }

#if 0
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  //  double * vj = static_cast<double*>(info_vj.ptr);
  
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  //  double * vk = static_cast<double*>(info_vk.ptr);
  printf("LIBGPU:: blksize= %i  naux= %i  nao= %i  nset= %i  count= %i\n",blksize,naux,nao,nset,count);
  printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  	 info_dmtril.shape[0], info_dmtril.shape[1],
  	 info_eri1.shape[0], info_eri1.shape[1],
  	 info_dmtril.shape[0], info_eri1.shape[0],
  	 info_vj.shape[0], info_vj.shape[1],
  	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);
  
  DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair);
  //  printf("LIBGPU:: eri1= %p  dfobj= %lu  count= %i  combined= %lu\n",eri1,addr_dfobj,count,addr_dfobj+count);
  printf("LIBGPU:: dfobj= %lu  count= %i  combined= %lu\n",addr_dfobj,count,addr_dfobj+count);
  printf("LIBGPU::     0:      %f %f %f %f\n",da_eri1(0,0), da_eri1(0,1), da_eri1(0,nao_pair-2), da_eri1(0,nao_pair-1));
  printf("LIBGPU::     1:      %f %f %f %f\n",da_eri1(1,0), da_eri1(1,1), da_eri1(1,nao_pair-2), da_eri1(1,nao_pair-1));
  printf("LIBGPU::     naux-2: %f %f %f %f\n",da_eri1(naux-2,0), da_eri1(naux-2,1), da_eri1(naux-2,nao_pair-2), da_eri1(naux-2,nao_pair-1));
  printf("LIBGPU::     naux-1: %f %f %f %f\n",da_eri1(naux-1,0), da_eri1(naux-1,1), da_eri1(naux-1,nao_pair-2), da_eri1(naux-1,nao_pair-1));
#endif

#ifdef _USE_ERI_CACHE
  // retrieve or cache eri block
  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count) {
      id = i;
      break;
    }

  if(id < eri_list.size()) {
    eri_count[id]++;

    int diff_size = eri_size[id] - naux * nao_pair;
    if(diff_size != 0) {
      printf("LIBGPU:: Error: eri_cache size != 0  diff_size= %i\n",diff_size);
      exit(1);
    }

    d_eri = d_eri_cache[id];
    double * eri_host = d_eri_host[id];
    double diff_eri = sqrt( (eri_host[0] + eri_host[1] - eri1[0] - eri1[nao_pair]) * (eri_host[0] + eri_host[1] - eri1[0] - eri1[nao_pair]) ); // this is dangerous; need something better
    if(diff_eri > 1e-10) {
      eri_host[0] = eri1[0];
      eri_host[1] = eri1[nao_pair];
      pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), stream);
      //      for(int i=0; i<naux * nao_pair; ++i) eri_[i] = eri1[i];
      eri_update[id]++;
    }

#if 0
    // debug eri cache on host
    double diffsq = 0.0;
    for(int i=0; i<naux * nao_pair; ++i) diffsq += (eri_[i] - eri1[i]) * (eri_[i] - eri1[i]);
    double diff = sqrt(diffsq);
    printf("LIBGPU:: eri_cache diff= %f\n",diff);
    if(diff > 1e-10) {
      printf("LIBGPU:: Error: eri_cache diff > TOL :: diff= %f\n",diff);
      exit(1);
    }
#endif
    
  }
  else
    {
      eri_list.push_back(addr_dfobj+count);
      eri_count.push_back(1);
      eri_update.push_back(0);
      eri_size.push_back(naux * nao_pair);

      d_eri_cache.push_back( (double *) pm->dev_malloc(naux * nao_pair * sizeof(double)) );
      int id = d_eri_cache.size() - 1;
      d_eri = d_eri_cache[ id ];
      
      pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), stream);
      //      for(int i=0; i<naux * nao_pair; ++i) eri_[i] = eri1[i];

      d_eri_host.push_back( (double *) pm->dev_malloc_host(2 * sizeof(double)) );
      double * d_eri_host_ = d_eri_host[id];
      d_eri_host_[0] = eri1[0];
      d_eri_host_[1] = eri1[nao_pair];
    }
#endif

#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif

#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array_jk[2] += t1 - t0;
#endif
  
  if(with_j) {
    
#ifdef _CUDA_NVTX
    nvtxRangePushA("GetJK::RHO+J");
#endif
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)
    {
      dim3 grid_size(nset, (naux + (_RHO_BLOCK_SIZE - 1)) / _RHO_BLOCK_SIZE, 1);
      dim3 block_size(1, _RHO_BLOCK_SIZE, 1);
      
      _getjk_rho<<<grid_size, block_size, 0, stream>>>(d_rho, d_dmtril, d_eri, nset, naux, nao_pair);
    }
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    {
      dim3 grid_size(nset, (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, 1);
      dim3 block_size(1, _DOT_BLOCK_SIZE, 1);
      
      _getjk_vj<<<grid_size, block_size, 0, stream>>>(d_vj, d_rho, d_eri, nset, nao_pair, naux, count);
    }
    
#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
    
#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
    t_array_jk[3] += t2 - t1;
#endif
  }
    
  // buf2 = lib.unpack_tril(eri1, out=buf[1])

  
#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
#endif
    
#ifdef _CUDA_NVTX
    nvtxRangePushA("GetJK::TRIL_MAP");
#endif
    
  {    
    dim3 grid_size((naux + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, (nao*nao + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, 1);
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    
    _getjk_unpack_buf2<<<grid_size, block_size, 0, stream>>>(d_buf2, d_eri, d_tril_map, naux, nao, nao_pair);
  }
    
#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
    
#ifdef _SIMPLE_TIMER
    double t3 = omp_get_wtime();
    t_array_jk[4] += t3 - t2;
#endif
    
  for(int indxK=0; indxK<nset; ++indxK) {

#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
#endif
    
#ifdef _CUDA_NVTX
    nvtxRangePushA("GetJK::Transfer DMS");
#endif
    
    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    double * dms = static_cast<double*>(info_dms.ptr);

    pm->dev_push_async(d_dms, dms, nao*nao*sizeof(double), stream);
    
#ifdef _CUDA_NVTX
    nvtxRangePop();
    nvtxRangePushA("GetJK::Batched DGEMM");
#endif

#ifdef _SIMPLE_TIMER
    double t5 = omp_get_wtime();
    t_array_jk[5] += t5 - t4;
#endif
    
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int nao2 = nao * nao;
      
      cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, nao, nao, nao,
				&alpha, d_buf2, nao, nao2,
				d_dms, nao, 0,
				&beta, d_buf1, nao, nao2, naux);
    }
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)

#ifdef _CUDA_NVTX
    nvtxRangePop();
    nvtxRangePushA("GetJK::Transpose");
#endif

#ifdef _SIMPLE_TIMER
    double t6 = omp_get_wtime();
    t_array_jk[6] += t6 - t5;
#endif
    
    {
      dim3 grid_size((naux + (_TRANSPOSE_BLOCK_SIZE - 1)) / _TRANSPOSE_BLOCK_SIZE,
		     (nao + (_TRANSPOSE_BLOCK_SIZE - 1)) / _TRANSPOSE_BLOCK_SIZE, 1);
      dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_BLOCK_SIZE, 1);
      
      _getjk_transpose_buf1_buf3<<<grid_size, block_size, 0, stream>>>(d_buf3, d_buf1, naux, nao);
    }
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = 1.0 * A.B + 0.0 * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T
    
    const double alpha = 1.0;
    const double beta = (count == 0) ? 0.0 : 1.0;
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = nao;
    
    const int vk_offset = indxK * nao*nao;
    
#ifdef _CUDA_NVTX
    nvtxRangePop();
    nvtxRangePushA("DGEMM");
#endif
    
#ifdef _SIMPLE_TIMER
    double t7 = omp_get_wtime();
    t_array_jk[7] += t7 - t6;
#endif
    
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_buf2, ldb, d_buf3, lda, &beta, d_vkk+vk_offset, ldc);

#ifdef _CUDA_NVTX
    nvtxRangePop();
    nvtxRangePushA("SYNC");
#endif
    
#ifdef _SIMPLE_TIMER
    double t8 = omp_get_wtime();
    t_array_jk[8] += t8 - t7;
#endif
    
    pm->dev_stream_wait(stream);
    
#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
   
#ifdef _SIMPLE_TIMER
    double t9 = omp_get_wtime();
    t_array_jk[9] += t9 - t8;
    t_array_jk[10] += t9 - t0;
    t_array_jk_count++;
#endif 
  }
  
  //  printf("Leaving get_jk()\n");
}
  
/* ---------------------------------------------------------------------- */

void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf)
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

__global__ void _hessop_get_veff_vk_1(double * vPpj, double * bPpj, double * vk_bj, int nvirt, int nocc, int naux, int ncore, int nmo)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nvirt) return;
  if(j >= nocc) return;

  double tmp = 0.0;
  for(int k=0; k<naux; ++k)
    for(int l=0; l<nocc; ++l)
      tmp += vPpj[k*nmo*nocc + (ncore+i)*nocc + l] * bPpj[k*nmo*nocc + l*nocc + j];
  vk_bj[i*nocc + j] = tmp;

}

/* ---------------------------------------------------------------------- */

__global__ void _hessop_get_veff_vk_2(double * vPpj, double * bPpj, double * vk_bj, int nvirt, int nocc, int naux, int ncore, int nmo)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nvirt) return;
  if(j >= nocc) return;

  double tmp = 0.0;
  for(int k=0; k<naux; ++k)
    for(int l=0; l<ncore; ++l)
      tmp += bPpj[k*nmo*nocc + (ncore+i)*nocc + l] * vPpj[k*nmo*nocc + j*nocc + l];
  vk_bj[i*nocc + j] += tmp;

}

/* ---------------------------------------------------------------------- */

void Device::hessop_get_veff(int naux, int nmo, int ncore, int nocc,
		    py::array_t<double> _bPpj, py::array_t<double> _vPpj, py::array_t<double> _vk_bj)
{
  py::buffer_info info_bPpj = _bPpj.request(); // 3D array (naux, nmo, nocc) : read-only
  py::buffer_info info_vPpj = _vPpj.request(); // 3D array (naux, nmo, nocc) : read-only 
  py::buffer_info info_vk_bj = _vk_bj.request(); // 2D array (nmo-ncore, nocc) : accumulate
  
  double * bPpj = static_cast<double*>(info_bPpj.ptr);
  double * vPpj = static_cast<double*>(info_vPpj.ptr);
  double * vk_bj = static_cast<double*>(info_vk_bj.ptr);
  
  int nvirt = nmo - ncore;

#if 0
  printf("LIBGPU:: naux= %i  nmo= %i  ncore= %i  nocc= %i  nvirt= %i\n",naux, nmo, ncore, nocc, nvirt);
  printf("LIBGPU:: shape : bPpj= (%i, %i, %i)  vPj= (%i, %i, %i)  vk_bj= (%i, %i)\n",
	 info_bPpj.shape[0],info_bPpj.shape[1],info_bPpj.shape[2],
	 info_vPpj.shape[0],info_vPpj.shape[1],info_vPpj.shape[2],
	 info_vk_bj.shape[0], info_vk_bj.shape[1]);
#endif
  
  DevArray3D da_bPpj = DevArray3D(bPpj, naux, nmo, nocc);
  DevArray3D da_vPpj = DevArray3D(vPpj, naux, nmo, nocc);
  DevArray2D da_vk_bj = DevArray2D(vk_bj, nvirt, nocc);

  int _size_bPpj = naux * nmo * nocc;
  if(_size_bPpj > size_bPpj) {
    size_bPpj = _size_bPpj;
    if(d_bPpj) pm->dev_free(d_bPpj);
    d_bPpj = (double *) pm->dev_malloc(size_bPpj * sizeof(double));
  }

  int _size_vPpj = naux * nmo * nocc;
  if(_size_vPpj > size_vPpj) {
    size_vPpj = _size_vPpj;
    if(d_vPpj) pm->dev_free(d_vPpj);
    d_vPpj = (double *) pm->dev_malloc(size_vPpj * sizeof(double));
  }
  
  int _size_vk_bj = (nmo-ncore) * nocc;
  if(_size_vk_bj > size_vk_bj) {
    size_vk_bj = _size_vk_bj;
    if(d_vk_bj) pm->dev_free(d_vk_bj);
    d_vk_bj = (double *) pm->dev_malloc(size_vk_bj * sizeof(double));
  }

  pm->dev_push_async(d_bPpj, bPpj, _size_bPpj*sizeof(double), stream);
  pm->dev_push_async(d_vPpj, vPpj, _size_vPpj*sizeof(double), stream);
  //  pm->dev_push_async(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double), stream);
  
  
  // vk_mo (bb|jj) in microcycle
  // vPbj = vPpj[:,ncore:,:] #np.dot (self.bPpq[:,ncore:,ncore:], dm_ai)
  // vk_bj = np.tensordot (vPbj, self.bPpj[:,:nocc,:], axes=((0,2),(0,1)))

#if 1

#ifdef _CUDA_NVTX
  nvtxRangePushA("HessOP_get_veff_vk_1");
#endif

  // placeholder... really need to reorder to expose more parallelism and improve read-access
  {
    dim3 grid_size( (nvirt + (_DEFAULT_BLOCK_SIZE - 1)) / _DEFAULT_BLOCK_SIZE, (nocc + (_DEFAULT_BLOCK_SIZE - 1)) / _DEFAULT_BLOCK_SIZE, 1);
    dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);

    _hessop_get_veff_vk_1<<<grid_size, block_size, 0, stream>>>(d_vPpj, d_bPpj, d_vk_bj, nvirt, nocc, naux, ncore, nmo);
  }
  
#ifdef _CUDA_NVTX
  nvtxRangePop();
#endif

  // pm->dev_pull_async(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double), stream);
  // pm->dev_stream_wait(stream);
  
#else

#pragma omp parallel for collapse(2)
  for(int i=0; i<nvirt; ++i)
    for(int j=0; j<nocc; ++j) {

      double tmp = 0.0;
      for(int k=0; k<naux; ++k)
	for(int l=0; l<nocc; ++l)
	  tmp += da_vPpj(k,ncore+i,l) * da_bPpj(k,l,j);
      da_vk_bj(i,j) = tmp;
      
    }
  
#endif
  
  // vk_mo (bi|aj) in microcycle
  // vPji = vPpj[:,:nocc,:ncore]
  // bPbi = self.bPpj[:,ncore:,:ncore]
  // vk_bj += np.tensordot (bPbi, vPji, axes=((0,2),(0,2)))

#if 1
  
#ifdef _CUDA_NVTX
  nvtxRangePushA("HessOP_get_veff_vk_2");
#endif

  // placeholder... really need to reorder to expose more parallelism and improve read-access
  {
    dim3 grid_size( (nvirt + (_DEFAULT_BLOCK_SIZE - 1)) / _DEFAULT_BLOCK_SIZE, (nocc + (_DEFAULT_BLOCK_SIZE - 1)) / _DEFAULT_BLOCK_SIZE, 1);
    dim3 block_size(_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE, 1);

    _hessop_get_veff_vk_2<<<grid_size, block_size, 0, stream>>>(d_vPpj, d_bPpj, d_vk_bj, nvirt, nocc, naux, ncore, nmo);
  }
  
#ifdef _CUDA_NVTX
  nvtxRangePop();
#endif

  pm->dev_pull_async(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double), stream);
  pm->dev_stream_wait(stream);
  
#else
  
#pragma omp parallel for collapse(2)
  for(int i=0; i<nvirt; ++i)
    for(int j=0; j<nocc; ++j) {
   
      double tmp = 0.0;
      for(int k=0; k<naux; ++k)
	for(int l=0; l<ncore; ++l)
	  tmp += da_bPpj(k,ncore+i,l) * da_vPpj(k,j,l);
      da_vk_bj(i,j) += tmp;
    }
  
#endif
  
}

#endif
