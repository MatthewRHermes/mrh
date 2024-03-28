/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _TRANSPOSE_BLOCK_SIZE 16
#define _TRANSPOSE_NUM_ROWS 16
#define _UNPACK_BLOCK_SIZE 32

#define _HESSOP_BLOCK_SIZE 32
#define _DEFAULT_BLOCK_SIZE 32

//#define _DEBUG_DEVICE

#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int _nset, int _nao, int _naux, int count)
{
#ifdef _DEBUG_DEVICE
  printf("Inside Device::init_get_jk()\n");
#endif
  
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
  
#ifdef _DEBUG_DEVICE
  printf(" -- Leaving Device::init_get_jk()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf(" -- Inside Device::pull_get_jk()\n");
#endif
  
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  
  double * vj = static_cast<double*>(info_vj.ptr);
 
  pm->dev_pull(d_vj, vj, nset * nao_pair * sizeof(double));

  if(with_k) {
    py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
    
    double * vk = static_cast<double*>(info_vk.ptr);
    
    pm->dev_pull(d_vkk, vk, nset * nao * nao * sizeof(double));
  }
  
  update_dfobj = 0;
  
#ifdef _DEBUG_DEVICE
  printf(" -- Leaving Device::pull_get_jk()\n");
#endif
}

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

#if 0

__global__ void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= naux) return;
  
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  while (j < nao*nao) {
    buf[j] = tril[ map[j] ];
    j += blockDim.y;  // * gridDim.y; // gridDim.y is just 1
  }
  
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
  //  for(int j=0; j<nao*nao; ++j) buf[j] = tril[ map[j] ];
}
#endif

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

/* ---------------------------------------------------------------------- */

// The _vj and _vk arguements aren't actually used anymore and could be removed. 
void Device::get_jk(int naux,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int with_k, int count, size_t addr_dfobj)
{
#ifdef _DEBUG_DEVICE
  printf("Inside Device::get_jk()\n");
#endif
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

#ifdef _CUDA_NVTX
    nvtxRangePushA("GetJK::Init");
#endif
  
  const int with_j = 1;
  
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
    if(d_rho) pm->dev_free(d_rho);
    d_rho = (double *) pm->dev_malloc(size_rho * sizeof(double));
  }

#if 0
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  //  double * vj = static_cast<double*>(info_vj.ptr);
  
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  //  double * vk = static_cast<double*>(info_vk.ptr);
  printf("LIBGPU:: blksize= %i  naux= %i  nao= %i  nset= %i  nao_pair= %i  count= %i\n",blksize,naux,nao,nset,nao_pair,count);
  printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  	 info_dmtril.shape[0], info_dmtril.shape[1],
  	 info_eri1.shape[0], info_eri1.shape[1],
  	 info_dmtril.shape[0], info_eri1.shape[0],
  	 info_vj.shape[0], info_vj.shape[1],
  	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);
  
  DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair);
  //  printf("LIBGPU:: eri1= %p  dfobj= %lu  count= %i  combined= %lu\n",eri1,addr_dfobj,count,addr_dfobj+count);
  printf("LIBGPU:: dfobj= %#012x  count= %i  combined= %#012x  update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count, update_dfobj);
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
    d_eri = d_eri_cache[id];
    
#ifdef _DEBUG_ERI_CACHE
    int diff_size = eri_size[id] - naux * nao_pair;
    if(diff_size != 0) {
      printf("LIBGPU:: Error: eri_cache size != 0  diff_size= %i\n",diff_size);
      exit(1);
    }
    
    double * eri_host = d_eri_host[id];
    double diff_eri = 0.0;
    for(int i=0; i<naux*nao_pair; ++i) diff_eri += (eri_host[i] - eri1[i]) * (eri_host[i] - eri1[i]);
    
    if(diff_eri > 1e-10) {
      for(int i=0; i<naux*nao_pair; ++i) eri_host[i] = eri1[i];
      pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), stream);
      eri_update[id]++;
      
      // update_dfobj fails to correctly update device ; this is an error
      if(!update_dfobj) {
 	printf("LIBGPU:: Warning: ERI updated on device w/ diff_eri= %f, but update_dfobj= %i\n",diff_eri,update_dfobj);
 	//count = -1;
 	//return;
 	exit(1);
      }
    } else {
      
      // update_dfobj falsely updates device ; this is loss of performance
      if(update_dfobj) {
 	printf("LIBGPU:: Warning: ERI not updated on device w/ diff_eri= %f, but update_dfobj= %i\n",diff_eri,update_dfobj);
 	//count = -1;
 	//return;
 	exit(1);
      }
    }
#else
    if(update_dfobj) {
      eri_update[id]++;
      pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), stream);
    }
#endif
    
  } else {
    eri_list.push_back(addr_dfobj+count);
    eri_count.push_back(1);
    eri_update.push_back(0);
    eri_size.push_back(naux * nao_pair);
    
    d_eri_cache.push_back( (double *) pm->dev_malloc(naux * nao_pair * sizeof(double)) );
    int id = d_eri_cache.size() - 1;
    d_eri = d_eri_cache[ id ];
    
    pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), stream);
    
#ifdef _DEBUG_ERI_CACHE
    d_eri_host.push_back( (double *) pm->dev_malloc_host(naux*nao_pair * sizeof(double)) );
    double * d_eri_host_ = d_eri_host[id];
    for(int i=0; i<naux*nao_pair; ++i) d_eri_host_[i] = eri1[i];
#endif
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
#if 1
      dim3 grid_size(nset, naux, 1);
      dim3 block_size(1, 1, _RHO_BLOCK_SIZE);
#else
      dim3 grid_size(nset, (naux + (_RHO_BLOCK_SIZE - 1)) / _RHO_BLOCK_SIZE, 1);
      dim3 block_size(1, _RHO_BLOCK_SIZE, 1);
#endif

      //      printf(" -- calling _getjk_rho()\n");
      _getjk_rho<<<grid_size, block_size, 0, stream>>>(d_rho, d_dmtril, d_eri, nset, naux, nao_pair);
    }
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    {
      dim3 grid_size(nset, (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, 1);
      dim3 block_size(1, _DOT_BLOCK_SIZE, 1);
      
      //      printf(" -- calling _getjk_vj()\n");
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

  if(!with_k) return;
  
  // buf2 = lib.unpack_tril(eri1, out=buf[1])
  
#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
#endif
    
#ifdef _CUDA_NVTX
    nvtxRangePushA("GetJK::TRIL_MAP");
#endif
    
    {
#if 0
      dim3 grid_size(naux, 1, 1);
      dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#else
      dim3 grid_size(naux, (nao*nao + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, 1);
      dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
      // dim3 grid_size((naux + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, (nao*nao + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, 1);
      // dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
#endif
      
      //    printf(" -- calling _unpack_buf2()\n");
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

    //    printf(" -- calling _dev_push_async(dms)\n");
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
#if 1
      dim3 grid_size( _TILE(naux*nao, _TRANSPOSE_BLOCK_SIZE), _TILE(nao, _TRANSPOSE_BLOCK_SIZE), 1);
      dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS);
      
      _transpose<<<grid_size, block_size, 0, stream>>>(d_buf3, d_buf1, naux*nao, nao);
#else
      dim3 grid_size(naux*nao, 1, 1);
      dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
      
      _transpose<<<grid_size, block_size, 0, stream>>>(d_buf3, d_buf1, naux*nao, nao);
#endif
      
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

    //    printf(" -- calling dev_stream_wait()\n");
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
  
#ifdef _DEBUG_DEVICE
  printf(" -- Leaving Device::get_jk()\n");
#endif
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

__global__ void _hessop_get_veff_reshape1(double * vPpj, double * buf, int nmo, int nocc, int ncore, int nvirt, int naux)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  int k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if(i >= nvirt) return;
  if(j >= nocc) return;

  // thread-local work

  const int indx1 = i*nocc*naux + j*naux;
  const int indx2 = (ncore+i)*nocc + j;
  
  while (k < naux) {
    vPpj[indx1 + k] = buf[k * nmo*nocc + indx2];
    k += blockDim.z; // * gridDim.z; // gridDim.z is just 1
  }
}

/* ---------------------------------------------------------------------- */

__global__ void _hessop_get_veff_reshape2(double * bPpj, double * buf, int nmo, int nocc, int ncore, int nvirt, int naux)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if(i >= nocc) return;
  if(j >= naux) return;

  // thread-local work

  const int indx1 = i*naux*nocc + j*nocc;
  const int indx2 = j*nmo*nocc + i*nocc;
  
  while (k < nocc) {
    bPpj[indx1 + k] = buf[indx2 + k];
    k += blockDim.z; // * gridDim.z; // gridDim.z is just 1
  }
}

/* ---------------------------------------------------------------------- */

__global__ void _hessop_get_veff_reshape3(double * bPpj, double * buf, int nmo, int nocc, int ncore, int nvirt, int naux)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if(i >= nvirt) return;
  if(j >= naux) return;

  // thread-local work

  const int indx1 = i*naux*ncore + j*ncore;
  const int indx2 = j*nmo*nocc + (ncore+i)*nocc;
  
  while (k < ncore) {
    bPpj[indx1 + k] = buf[indx2 + k];
    k += blockDim.z; // * gridDim.z; // gridDim.z is just 1
  }
}

/* ---------------------------------------------------------------------- */

__global__ void _hessop_get_veff_reshape4(double * vPpj, double * buf, int nmo, int nocc, int ncore, int nvirt, int naux)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  int k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if(i >= naux) return;
  if(j >= ncore) return;

  // thread-local work

  const int indx1 = i*ncore*nocc + j*nocc;
  const int indx2 = i*nmo*nocc + j;
  
  while (k < nocc) {
    vPpj[indx1 + k] = buf[indx2 + k * nocc];
    k += blockDim.z; // * gridDim.z; // gridDim.z is just 1
  }
}

/* ---------------------------------------------------------------------- */

void Device::hessop_push_bPpj(py::array_t<double> _bPpj)
{
  py::buffer_info info_bPpj = _bPpj.request(); // 3D array (naux, nmo, nocc) : read-only

  double * bPpj = static_cast<double*>(info_bPpj.ptr);

#if 0
  printf("LIBGPU:: naux= %i  nmo= %i  nocc= %i\n",naux, nmo, nocc);
  printf("LIBGPU:: hessop_push_bPpj : bPpj= (%i, %i, %i)\n",
	 info_bPpj.shape[0],info_bPpj.shape[1],info_bPpj.shape[2]);

  DevArray3D da_bPpj = DevArray3D(bPpj, naux, nmo, nocc);
  printf("LIBGPU :: hessop_push_bPpj :: bPpj= %f %f %f %f\n", da_bPpj(0,0,0), da_bPpj(0,1,0), da_bPpj(1,0,0), da_bPpj(naux-1,0,0));
#endif

  // really need to clean up initializing naux, nmo, nocc, ncore, etc... if really constants for duration of calculation

  int _naux = info_bPpj.shape[0];
  int _nmo = info_bPpj.shape[1];
  int _nocc = info_bPpj.shape[2];

  // this should all be wrapped in a new pm->dev_smalloc() w/ a very simple struct

  int _size_bPpj = _naux * _nmo * _nocc;
  if(_size_bPpj > size_bPpj) {
    size_bPpj = _size_bPpj;
    if(d_bPpj) pm->dev_free(d_bPpj);
    d_bPpj = (double *) pm->dev_malloc(size_bPpj * sizeof(double));
  }

  pm->dev_push_async(d_bPpj, bPpj, _size_bPpj*sizeof(double), stream);
}

/* ---------------------------------------------------------------------- */

void Device::hessop_get_veff(int naux, int nmo, int ncore, int nocc,
		    py::array_t<double> _bPpj, py::array_t<double> _vPpj, py::array_t<double> _vk_bj)
{
  py::buffer_info info_vPpj = _vPpj.request(); // 3D array (naux, nmo, nocc) : read-only 
  py::buffer_info info_vk_bj = _vk_bj.request(); // 2D array (nmo-ncore, nocc) : accumulate
  
  double * vPpj = static_cast<double*>(info_vPpj.ptr);
  double * vk_bj = static_cast<double*>(info_vk_bj.ptr);
  
  int nvirt = nmo - ncore;

#if 0
  py::buffer_info info_bPpj = _bPpj.request(); // 3D array (naux, nmo, nocc) : read-only
  double * bPpj = static_cast<double*>(info_bPpj.ptr);
  
  printf("LIBGPU:: naux= %i  nmo= %i  ncore= %i  nocc= %i  nvirt= %i\n",naux, nmo, ncore, nocc, nvirt);
  printf("LIBGPU:: shape : bPpj= (%i, %i, %i)  vPj= (%i, %i, %i)  vk_bj= (%i, %i)\n",
	 info_bPpj.shape[0],info_bPpj.shape[1],info_bPpj.shape[2],
	 info_vPpj.shape[0],info_vPpj.shape[1],info_vPpj.shape[2],
	 info_vk_bj.shape[0], info_vk_bj.shape[1]);
#endif

  // this buf realloc needs to be consistent with that in init_get_jk()
  
  int _size_buf = naux * nmo * nocc;
  if(_size_buf > size_buf) {
    size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double*) pm->dev_malloc_host(2*size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(size_buf*sizeof(double));
    buf4 = (double *) pm->dev_malloc_host(size_buf*sizeof(double));

    if(d_buf1) pm->dev_free(d_buf1);
    if(d_buf2) pm->dev_free(d_buf2);
    if(d_buf3) pm->dev_free(d_buf3);
    
    d_buf1 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf2 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf3 = (double *) pm->dev_malloc(size_buf * sizeof(double));
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
  
  // vk_mo (bb|jj) in microcycle
  // vPbj = vPpj[:,ncore:,:] #np.dot (self.bPpq[:,ncore:,ncore:], dm_ai)
  // vk_bj = np.tensordot (vPbj, self.bPpj[:,:nocc,:], axes=((0,2),(0,1)))
  
#if 1
  pm->dev_push_async(d_buf1, vPpj, naux*nmo*nocc*sizeof(double), stream);

  {
    dim3 grid_size(nvirt, nocc, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
    
    _hessop_get_veff_reshape1<<<grid_size, block_size, 0, stream>>>(d_vPpj, d_buf1, nmo, nocc, ncore, nvirt, naux);
  }

  {
    dim3 grid_size(nocc, naux, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
    
    _hessop_get_veff_reshape2<<<grid_size, block_size, 0, stream>>>(d_buf2, d_bPpj, nmo, nocc, ncore, nvirt, naux);
  }

  {
    const double alpha = 1.0;
    const double beta = 0.0;
    
    const int m = nocc; // # of rows in first matrix
    const int n = nvirt; // # of columns in second matrix
    const int k = naux*nocc; // # of columns in first matrix
    
    const int lda = nocc;
    const int ldb = nocc*naux;
    const int ldc = nocc;
    
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_buf2, lda, d_vPpj, ldb, &beta, d_vk_bj, ldc);
  }
  
#else

  double * buf_vPpj = buf_tmp;
  double * buf_bPpj = buf3;
  
#pragma omp parallel for collapse(2)
  for(int i=0; i<nvirt; ++i)
    for(int j=0; j<nocc; ++j) {
      
      const int indx1 = i*nocc*naux + j*naux;
      const int indx2 = (ncore+i)*nocc + j;
      for(int k=0; k<naux; ++k) 
	buf_vPpj[indx1 + k] = vPpj[k * nmo*nocc + indx2]; // (nvirt, nocc, naux)
      
    }

#pragma omp parallel for collapse(2)
  for(int i=0; i<nocc; ++i)
    for(int j=0; j<naux; ++j) {

      const int indx1 = i*naux*nocc + j*nocc;
      const int indx2 = j*nmo*nocc + i*nocc;
      for(int k=0; k<nocc; ++k)
	buf_bPpj[indx1 + k] = bPpj[indx2 + k];
    }

  // To compute A.B w/ Fortran ordering, we ask for B.A as if B and A were transposed
  // Computing A.B, where A = vPpj and B = bPpj
  // Ask for A=bPpj, B= vPpj, m= # columns of bPpj, n= # rows of vPpj, k= # rows of bPpj

  {
    const double alpha = 1.0;
    const double beta = 0.0;
    
    const int m = nocc; // # of rows in first matrix
    const int n = nvirt; // # of columns in second matrix
    const int k = naux*nocc; // # of columns in first matrix
    
    const int lda = nocc;
    const int ldb = nocc*naux;
    const int ldc = nocc;
    
    dgemm_((char *) "N", (char *) "N", &m, &n, &k, &alpha, buf_bPpj, &lda, buf_vPpj, &ldb, &beta, vk_bj, &ldc);
  }
  
#endif
  
  // vk_mo (bi|aj) in microcycle
  // vPji = vPpj[:,:nocc,:ncore]
  // bPbi = self.bPpj[:,ncore:,:ncore]
  // vk_bj += np.tensordot (bPbi, vPji, axes=((0,2),(0,2)))

#if 1
  {
    dim3 grid_size(nvirt, naux, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
    
    _hessop_get_veff_reshape3<<<grid_size, block_size, 0, stream>>>(d_buf2, d_bPpj, nmo, nocc, ncore, nvirt, naux);
  }
  
  {
    dim3 grid_size(naux, ncore, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
    
    _hessop_get_veff_reshape4<<<grid_size, block_size, 0, stream>>>(d_vPpj, d_buf1, nmo, nocc, ncore, nvirt, naux);
  }


  {
    const double alpha = 1.0;
    const double beta = 1.0;
    
    const int m = nocc; // # of rows in first matrix
    const int n = nvirt; // # of columns in second matrix
    const int k = naux*ncore; // # of columns in first matrix
    
    const int lda = nocc;
    const int ldb = ncore*naux;
    const int ldc = nocc;
    
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_vPpj, lda, d_buf2, ldb, &beta, d_vk_bj, ldc);
  }

  pm->dev_stream_wait(stream);
  pm->dev_pull(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double));
    
#else
  
#pragma omp parallel for collapse(2)
  for(int i=0; i<nvirt; ++i)
    for(int j=0; j<naux; ++j) {
      
      const int indx1 = i*naux*ncore + j*ncore;
      const int indx2 = j*nmo*nocc + (ncore+i)*nocc;
      for(int k=0; k<ncore; ++k)
	buf_bPpj[indx1 + k] = bPpj[indx2 + k];
    }

#pragma omp parallel for collapse(2)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncore; ++j) {

      const int indx1 = i*ncore*nocc + j*nocc;
      const int indx2 = i*nmo*nocc + j;
      for(int k=0; k<nocc; ++k)
	buf_vPpj[indx1 + k] = vPpj[indx2 + k*nocc];

    }

  pm->dev_pull(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double));
  
  {
    const double alpha = 1.0;
    const double beta = 1.0;
    
    const int m = nocc; // # of rows in first matrix
    const int n = nvirt; // # of columns in second matrix
    const int k = naux*ncore; // # of columns in first matrix
    
    const int lda = nocc;
    const int ldb = ncore*naux;
    const int ldc = nocc;
    
    dgemm_((char *) "N", (char *) "N", &m, &n, &k, &alpha, buf_vPpj, &lda, buf_bPpj, &ldb, &beta, vk_bj, &ldc);
  }
#endif
  
}

#endif
