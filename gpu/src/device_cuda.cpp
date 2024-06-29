/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "device.h"

#include <stdio.h>
//#include <cuda_runtime_api.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _TRANSPOSE_BLOCK_SIZE 16
#define _TRANSPOSE_NUM_ROWS 16
#define _UNPACK_BLOCK_SIZE 32

#define _HESSOP_BLOCK_SIZE 32
#define _DEFAULT_BLOCK_SIZE 32

#define _DEBUG_DEVICE

#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int _nset, int _nao, int _naux, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::init_get_jk()\n");
#endif

  profile_start("init_get_jk");
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);
  
  blksize = _blksize;
  nset = _nset;
  nao = _nao;
  naux = _naux;

  nao_pair = nao * (nao+1) / 2;
  
  int _size_vj = nset * nao_pair;
  if(_size_vj > dd->size_vj) {
    dd->size_vj = _size_vj;
    if(dd->d_vj) pm->dev_free(dd->d_vj);
    dd->d_vj = (double *) pm->dev_malloc(_size_vj * sizeof(double));
  }
  
  int _size_vk = nset * nao * nao;
  if(_size_vk > dd->size_vk) {
    dd->size_vk = _size_vk;
    
    if(dd->d_vkk) pm->dev_free(dd->d_vkk);
    dd->d_vkk = (double *) pm->dev_malloc(_size_vk * sizeof(double));
  }

  int _size_buf = blksize * nao * nao;
  if(_size_buf > dd->size_buf) {
    dd->size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double *) pm->dev_malloc_host(2*_size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(_size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) pm->dev_malloc_host(_size_buf*sizeof(double)); // (blksize*nao, nao)

    if(dd->d_buf1) pm->dev_free(dd->d_buf1);
    if(dd->d_buf2) pm->dev_free(dd->d_buf2);
    if(dd->d_buf3) pm->dev_free(dd->d_buf3);
    
    dd->d_buf1 = (double *) pm->dev_malloc(_size_buf * sizeof(double));
    dd->d_buf2 = (double *) pm->dev_malloc(_size_buf * sizeof(double));
    dd->d_buf3 = (double *) pm->dev_malloc(_size_buf * sizeof(double));
  }
  
  int _size_dms = nset * nao * nao;
  if(_size_dms > dd->size_dms) {
    dd->size_dms = _size_dms;
    if(dd->d_dms) pm->dev_free(dd->d_dms);
    dd->d_dms = (double *) pm->dev_malloc(_size_dms * sizeof(double));
  }

  int _size_dmtril = nset * nao_pair;
  if(_size_dmtril > dd->size_dmtril) {
    dd->size_dmtril = _size_dmtril;
    if(dd->d_dmtril) pm->dev_free(dd->d_dmtril);
    dd->d_dmtril = (double *) pm->dev_malloc(_size_dmtril * sizeof(double));
  }
  
  int _size_eri1 = naux * nao_pair;
  if(_size_eri1 > dd->size_eri1) {
    dd->size_eri1 = _size_eri1;
    if(dd->d_eri1) pm->dev_free(dd->d_eri1);
    dd->d_eri1 = (double *) pm->dev_malloc(_size_eri1 * sizeof(double));
  }
  
  int _size_tril_map = nao * nao;

  auto it = std::find(dd->size_tril_map.begin(), dd->size_tril_map.end(), _size_tril_map);

  int indx = it - dd->size_tril_map.begin();

  if(indx == dd->size_tril_map.size()) {
    dd->size_tril_map.push_back(_size_tril_map);
    dd->tril_map.push_back(nullptr);
    dd->d_tril_map.push_back(nullptr);

    dd->tril_map[indx] = (int *) pm->dev_malloc_host(_size_tril_map * sizeof(int));
    dd->d_tril_map[indx] = (int *) pm->dev_malloc(_size_tril_map * sizeof(int));
    int _i, _j, _ij;
    int * tm = dd->tril_map[indx];
    for(_ij = 0, _i = 0; _i < nao; _i++)
      for(_j = 0; _j<=_i; _j++, _ij++) {
    	tm[_i*nao + _j] = _ij;
    	tm[_i + nao*_j] = _ij;
      }
    
    pm->dev_push(dd->d_tril_map[indx], dd->tril_map[indx], _size_tril_map*sizeof(int));
  }

  dd->d_tril_map_ptr = dd->d_tril_map[indx];

  int _size_buf_vj = num_devices * nset * nao_pair;
  if(_size_buf_vj > size_buf_vj) {
    size_buf_vj = _size_buf_vj;
    if(buf_vj) pm->dev_free_host(buf_vj);
    buf_vj = (double *) pm->dev_malloc_host(_size_buf_vj*sizeof(double));
  }

  int _size_buf_vk = num_devices * nset * nao * nao;
  if(_size_buf_vk > size_buf_vk) {
    size_buf_vk = _size_buf_vk;
    if(buf_vk) pm->dev_free_host(buf_vk);
    buf_vk = (double *) pm->dev_malloc_host(_size_buf_vk*sizeof(double));
  }

  // 1-time initialization
  
  // Create cuda stream
  
  if(dd->stream == nullptr) pm->dev_stream_create(dd->stream);
  
  // Create blas handle

  if(dd->handle == nullptr) {
    
#ifdef _DEBUG_DEVICE
    printf(" -- calling cublasCreate(&handle)\n");
#endif
    cublasCreate(&(dd->handle));
    _CUDA_CHECK_ERRORS();
#ifdef _DEBUG_DEVICE
    printf(" -- calling cublasSetStream(handle, stream)\n");
#endif
    cublasSetStream(dd->handle, dd->stream);
    _CUDA_CHECK_ERRORS();
  }
  
  profile_stop();
    
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[0] += t1 - t0;
#endif

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::init_get_jk()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int with_k)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_get_jk()\n");
#endif

#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
    
  profile_start("pull_get_jk");
  
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  
  double * vj = static_cast<double*>(info_vj.ptr);

  int size = nset * nao_pair * sizeof(double);

  double * tmp;
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);

    if(i == 0) tmp = vj;
    else tmp = &(buf_vj[i * nset * nao_pair]);
    
    if(dd->d_vj) pm->dev_pull_async(dd->d_vj, tmp, size, dd->stream);
  }
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pm->dev_stream_wait(dd->stream);

    if(i > 0 && dd->d_vj) {
      
      tmp = &(buf_vj[i * nset * nao_pair]);
#pragma omp parallel for
      for(int j=0; j<nset*nao_pair; ++j) vj[j] += tmp[j];
      
    }
  }

  update_dfobj = 0;
  if(!with_k) {
    profile_stop();
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
#endif
    
    return;
  }
    
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
    
  double * vk = static_cast<double*>(info_vk.ptr);

  size = nset * nao * nao * sizeof(double);

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
      
    my_device_data * dd = &(device_data[i]);

    if(i == 0) tmp = vk;
    else tmp = &(buf_vk[i * nset * nao * nao]);

    if(dd->d_vkk) pm->dev_pull_async(dd->d_vkk, tmp, size, dd->stream);
  }

  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pm->dev_stream_wait(dd->stream);

    if(i > 0 && dd->d_vkk) {
      
      tmp = &(buf_vk[i * nset * nao * nao]);
#pragma omp parallel for
      for(int j=0; j<nset*nao*nao; ++j) vk[j] += tmp[j];
    
    }

  }

  profile_stop();
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[1] += t1 - t0;
#endif
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::pull_get_jk()\n");
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
  printf("LIBGPU :: Inside Device::get_jk() w/ with_k= %i\n",with_k);
#endif
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  profile_start("get_jk :: init");

  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);
    
  const int with_j = 1;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, nao_pair)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int nao_pair = nao * (nao+1) / 2;

  double * d_eri;
  if(!use_eri_cache) {
    // if not caching, then eri block always transferred
    
    pm->dev_push_async(dd->d_eri1, eri1, naux * nao_pair * sizeof(double), dd->stream);
    d_eri = dd->d_eri1;
  }

  if(count < num_devices) {
    int err = pm->dev_push_async(dd->d_dmtril, dmtril, nset * nao_pair * sizeof(double), dd->stream);
    if(err) {
      printf("LIBGPU:: dev_push_async(d_dmtril) failed on count= %i\n",count);
      exit(1);
    }
  }
  
  int _size_rho = nset * naux;
  if(_size_rho > dd->size_rho) {
    dd->size_rho = _size_rho;
    if(dd->d_rho) pm->dev_free(dd->d_rho);
    dd->d_rho = (double *) pm->dev_malloc(_size_rho * sizeof(double));
  }

#if 0
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  printf("LIBGPU:: device= %i  blksize= %i  naux= %i  nao= %i  nset= %i  nao_pair= %i  count= %i\n",device_id,blksize,naux,nao,nset,nao_pair,count);
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
  
  if(use_eri_cache) {
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: Starting eri_cache lookup\n");
#endif

    // retrieve id of cached eri block

    int id = eri_list.size();
    for(int i=0; i<eri_list.size(); ++i)
      if(eri_list[i] == addr_dfobj+count) {
	id = i;
	break;
      }
    
    // grab/update cached data
    
    if(id < eri_list.size()) {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU :: -- eri block found: id= %i\n",id);
#endif
      
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
	pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), dd->stream);
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
#ifdef _DEBUG_DEVICE
	printf("LIBGPU :: -- updating eri block: id= %i\n",id);
#endif
	eri_update[id]++;
	int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), dd->stream);
	if(err) {
	  printf("LIBGPU:: dev_push_async(d_eri) updating eri block\n");
	  exit(1);
	}
      }
#endif
      
    } else {
      eri_list.push_back(addr_dfobj+count);
      eri_count.push_back(1);
      eri_update.push_back(0);
      eri_size.push_back(naux * nao_pair);
      eri_device.push_back(device_id);
      
      eri_num_blocks.push_back(0); // grow array
      eri_num_blocks[id-count]++;  // increment # of blocks for this dfobj
      
      eri_extra.push_back(naux);
      eri_extra.push_back(nao_pair);
      
      int id = d_eri_cache.size();
#ifdef _DEBUG_DEVICE
      printf("LIBGPU :: -- allocating new eri block: %i\n",id);
#endif
      
      d_eri_cache.push_back( (double *) pm->dev_malloc(naux * nao_pair * sizeof(double)) );
      d_eri = d_eri_cache[ id ];

#ifdef _DEBUG_DEVICE
      printf("LIBGPU :: -- initializing eri block\n");
#endif
      int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), dd->stream);
      if(err) {
	printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
	exit(1);
      }
      
#ifdef _DEBUG_ERI_CACHE
      d_eri_host.push_back( (double *) pm->dev_malloc_host(naux*nao_pair * sizeof(double)) );
      double * d_eri_host_ = d_eri_host[id];
      for(int i=0; i<naux*nao_pair; ++i) d_eri_host_[i] = eri1[i];
#endif
    }
  } // if(use_eri_cache)

  profile_stop();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting with_j calculation\n");
#endif
	 
  if(with_j) {
    
    profile_start("get_jk :: with_j");
    
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
      _getjk_rho<<<grid_size, block_size, 0, dd->stream>>>(dd->d_rho, dd->d_dmtril, d_eri, nset, naux, nao_pair);
    }
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    {
      dim3 grid_size(nset, (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, 1);
      dim3 block_size(1, _DOT_BLOCK_SIZE, 1);
      
      //      printf(" -- calling _getjk_vj()\n");
      int init = (count < num_devices) ? 1 : 0;
      _getjk_vj<<<grid_size, block_size, 0, dd->stream>>>(dd->d_vj, dd->d_rho, d_eri, nset, nao_pair, naux, init);
    }

    profile_stop();
  }

  if(!with_k) {
    
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array[2] += t1 - t0;
#endif
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- Leaving Device::get_jk()\n");
#endif
    
    return;
  }
  
  // buf2 = lib.unpack_tril(eri1, out=buf[1])
    
  profile_start("get_jk :: with_k");
    
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
    
    _getjk_unpack_buf2<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf2, d_eri, dd->d_tril_map_ptr, naux, nao, nao_pair);
  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: Starting with_k calculation\n");
#endif
  
  for(int indxK=0; indxK<nset; ++indxK) {

#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
#endif
    
    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    double * dms = static_cast<double*>(info_dms.ptr);

    double * d_dms = &(dd->d_dms[indxK*nao*nao]);

    if(count < num_devices) {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- calling dev_push_async(dms) for indxK= %i  nset= %i\n",indxK,nset);
#endif
    
      int err = pm->dev_push_async(d_dms, dms, nao*nao*sizeof(double), dd->stream);
      if(err) {
	printf("LIBGPU:: dev_push_async(d_dms) on indxK= %i\n",indxK);
	exit(1);
      }
    }
    
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int nao2 = nao * nao;

#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- calling cublasDgemmStrideBatched()\n");
#endif
      cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_T, CUBLAS_OP_T, nao, nao, nao,
				&alpha, dd->d_buf2, nao, nao2,
				d_dms, nao, 0,
				&beta, dd->d_buf1, nao, nao2, naux);
    }
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)
    
    {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- calling _transpose()\n");
#endif
      
#if 1
      dim3 grid_size( _TILE(naux*nao, _TRANSPOSE_BLOCK_SIZE), _TILE(nao, _TRANSPOSE_BLOCK_SIZE), 1);
      dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS);
      
      _transpose<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf3, dd->d_buf1, naux*nao, nao);
#else
      dim3 grid_size(naux*nao, 1, 1);
      dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
      
      _transpose<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf3, dd->d_buf1, naux*nao, nao);
#endif
      
    }
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = alpha * A.B + beta * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T
    
    const double alpha = 1.0;
    const double beta = (count < num_devices) ? 0.0 : 1.0; // first pass by each device initializes array, otherwise accumulate
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = nao;
    
    const int vk_offset = indxK * nao*nao;

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- calling cublasDgemm()\n");
#endif
    cublasDgemm(dd->handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dd->d_buf2, ldb, dd->d_buf3, lda, &beta, (dd->d_vkk)+vk_offset, ldc);
  }
  
  profile_stop();
    
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[2] += t1 - t0;
#endif
    
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- finished\n");
  printf("LIBGPU :: -- Leaving Device::get_jk()\n");
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
void Device::df_ao2mo_pass1_fdrv (int naux, int nmo, int nao, int blksize, 
			py::array_t<double> _bufpp, py::array_t<double> _mo_coeff, py::array_t<double> _eri1)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::df_ao2mo_pass1_fdrv()\n");
#endif
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  profile_start(" df_ao2mo_pass1_fdrv setup\n");
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  py::buffer_info info_bufpp = _bufpp.request(); // 3D array (naux,nmo,nmo)
  py::buffer_info info_mo_coeff = _mo_coeff.request(); // 2D array (nmo, nmo)
#ifdef _DEBUG_DEVICE
  printf("LIBGPU::: naux= %i  nmo= %i  nao= %i  blksize=%i \n",naux,nmo,nao,blksize);
  printf("LIBGPU::shape: _eri1= (%i,%i)  _mo_coeff= (%i,%i)  _bufpp= (%i, %i, %i)\n",
  	 info_eri1.shape[0], info_eri1.shape[1],
  	 info_mo_coeff.shape[0], info_mo_coeff.shape[1],
  	 info_bufpp.shape[0], info_bufpp.shape[1],info_bufpp.shape[2]);
#endif
  const int nao_pair = nao*(nao+1)/2;
  double * eri = static_cast<double*>(info_eri1.ptr);
  int _size_eri_unpacked = naux * nao * nao; 
  double * bufpp = static_cast<double*>(info_bufpp.ptr);
  int _size_bufpp = naux * nao * nao;
  double * mo_coeff = static_cast<double*>(info_mo_coeff.ptr);
  int _size_mo_coeff = nao * nao;
  const int device_id = 0;//count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
#ifdef _DEBUG_DEVICE
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#if 1 //for fastest
  {
  double * d_bufpp = (double*) pm->dev_malloc (sizeof(double)* _size_bufpp);//set memory for the entire bufpp array, no pushing needed 
  double * d_buf = (double*) pm->dev_malloc(_size_bufpp*sizeof(double)); //for eri*mo_coeff (don't pull or push)
  double * d_eri_unpacked = (double*) pm->dev_malloc (sizeof(double)* _size_eri_unpacked);//set memory for the entire eri array on GPU
  //unpack 2D eri of size naux * nao(nao+1)/2 to a full naux*nao*nao 3D matrix
  #if 0 //for unpacking
  {
  double * d_eri = (double*) pm->dev_malloc (sizeof(double)* _size_eri);//set memory for the entire eri array on GPU
  pm->dev_push(d_eri, eri, _size_eri * sizeof(double));//doing this allocation and pushing first because it doesn't change over iterations. 
  //int _size_tril_map = naux * nao * nao;
  int _size_tril_map = nao * nao;
  int * my_tril_map = (int*) malloc (_size_tril_map * sizeof(int));
  int * my_d_tril_map = (int*) pm->dev_malloc (_size_tril_map * sizeof(int));
  //for (int indx =0; indx<naux;++indx){
    int _i, _j, _ij;
    for(_ij = 0, _i = 0; _i < nao; ++_i)
      for(_j = 0; _j<=_i; ++_j, ++_ij) {
    	//my_tril_map[indx*nao*nao + _i*nao + _j] = _ij;
    	//my_tril_map[indx*nao*nao + _i + nao*_j] = _ij;
    	my_tril_map[_i*nao + _j] = _ij;
    	my_tril_map[_i + nao*_j] = _ij;
    } //}
  #ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- map created in ao2mo_fdrv\n");
  #endif
  int * my_d_tril_map_ptr=my_d_tril_map;
  pm->dev_push(my_d_tril_map,my_tril_map,_size_tril_map*sizeof(int));
  dim3 grid_size(naux, (nao*nao + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, 1);
  dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
  //_getjk_unpack_buf2<<<grid_size, block_size, 0, dd->stream>>>(eri1, d_eri_unpacked, dd->d_tril_map_ptr, naux, nao, nao_pair);
  _getjk_unpack_buf2<<<grid_size,block_size,0,dd->stream>>>(d_eri_unpacked, d_eri, my_d_tril_map_ptr, naux, nao, nao_pair);
  }
  #else  //for unpacking
  {
  double * h_eri_unpacked = (double*) malloc (sizeof(double)* _size_eri_unpacked);//set memory for the entire eri array on CPU side (temporary until we get getjk_unpack working)
  //#ifdef _DEBUG_DEVICE
  //double * h_eri_unpacked_reference = (double*) malloc (sizeof(double)* _size_eri_unpacked);//set memory for the entire eri array on CPU side for checking getjk unpack results
  //#endif
  for (int i = 0; i < naux; i++) {
    int _i, _j, _ij;
    double * tril = eri + nao_pair*i;
    for (_ij = 0, _i = 0; _i < nao; _i++){ 
      for (_j = 0; _j <= _i; _j++, _ij++){ 
        h_eri_unpacked[i*nao*nao+_i*nao+_j] = tril[_ij];
        h_eri_unpacked[i*nao*nao+_j*nao+_i] = tril[_ij];
  } } }
  pm->dev_push(d_eri_unpacked,h_eri_unpacked,_size_eri_unpacked*sizeof(double));
  }
  #endif //for unpacking
  //#ifdef _DEBUG_DEVICE
  //pm->dev_pull(h_eri_unpacked_reference, d_eri_unpacked, _size_eri_unpacked * sizeof(double));//push entire eri array
  //for(int aux=0;aux<1;++aux){for(int x1=0;x1<=nao;++x1){for(int x2=0;x2<=nao;++x2){printf("%f  ",h_eri_unpacked_reference[aux*nao*nao + x1*nao + x2]);}printf("\n");}printf("\n");}
  //for(int aux=0;aux<1;++aux){for(int x1=0;x1<=nao;++x1){for(int x2=0;x2<=nao;++x2){printf("%f  ",h_eri_unpacked[aux*nao*nao + x1*nao + x2]);}printf("\n");}printf("\n");}
  //#endif
  double *d_mo_coeff= (double*) pm->dev_malloc(_size_mo_coeff*sizeof(double));//allocate mo_coeff (might be able to avoid if already used in get_jk)
  pm->dev_push(d_mo_coeff, mo_coeff, _size_mo_coeff * sizeof(double));//doing this allocation and pushing first because it doesn't change over iterations. 
  double alpha = 1.0;
  double beta = 0.0;
  //bufpp = mo.T @ eri @ mo 
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- calling cublasDgemmStrideBatched() in ao2mo_fdrv\n");
#endif
  profile_stop();
  profile_start(" df_ao2mo_pass1_fdrv StridedBatchedDgemm\n");
  //buf = np.einsum('ijk,kl->ijl',eri_unpacked,mo_coeff),i=naux,j=nao,l=nao 
  cublasDgemmStridedBatched(dd->handle, 
                      CUBLAS_OP_N, CUBLAS_OP_N,nao, nao, nao, 
                      &alpha,d_eri_unpacked, nao, nao*nao,d_mo_coeff, nao, 0, 
                      &beta,d_buf, nao, nao*nao,naux);
  _CUDA_CHECK_ERRORS();
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- calling cublasDgemmStrideBatched() in ao2mo_fdrv \n");
#endif
  //bufpp = np.einsum('jk,ikl->ijl',mo_coeff.T,buf),i=naux,j=nao,l=nao 
  cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, nao, 
                      &alpha,d_mo_coeff, nao, 0, d_buf, nao, nao*nao, 
                      &beta, d_bufpp, nao, nao*nao,naux);
  _CUDA_CHECK_ERRORS();
  profile_stop();
  profile_start(" df_ao2mo_pass1_fdrv Data pull\n");
  pm->dev_pull(d_bufpp, bufpp, _size_bufpp * sizeof(double));
  pm->dev_free(d_mo_coeff);
  pm->dev_free(d_bufpp);
  pm->dev_free(d_buf);
  pm->dev_free(d_eri_unpacked);
  //pm->dev_free(d_eri);
  profile_stop();
  #ifdef _DEBUG_DEVICE
    printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
  #endif
  }
#else //for fastest
  {
  int _size_buf_bufpp = (nao)* (nao);
  int _size_buf = (nao)* (nao);
  int _size_buf_eri = (nao)* (nao);
  double *buf_eri = (double *) malloc(sizeof(double) * _size_buf_eri);//to convert eri slices to nao*nao (right now on cpu)
  double * d_buf_eri = (double*) pm->dev_malloc( _size_buf_eri*sizeof(double)); //for eri slices converted to nao*nao (push this) 
  double * d_buf = (double*) pm->dev_malloc(_size_buf*sizeof(double)); //for eri*mo_coeff (don't pull or push)
  double * h_buf = (double*) malloc(_size_buf*sizeof(double)); //for eri*mo_coeff (don't pull or push)
  double * d_buf_bufpp = (double*) pm->dev_malloc(_size_buf_bufpp*sizeof(double));  //for mo_coeff*d_buf (pull this)
  for (int i = 0; i < naux; i++) {
    const int ij_pair = nao*nao;//(*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);//ij_pair=nmo*nmo
    const int nao2 = nao*(nao+1)/2;//(*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);//
     //_getjk_unpack_buf2(double * buf_eri, double * eri1 + nao2*i, int * map, int naux, int nao, int nao_pair)
       int _i, _j, _ij;
       double * tril = eri1 + nao2*i;
       for (_ij = 0, _i = 0; _i < nao; _i++) 
         for (_j = 0; _j <= _i; _j++, _ij++) buf_eri[_i*nao+_j] = tril[_ij];
     //GPU code starts#if 1
     double alpha = 1.0;
     double beta = 0.0;
     //eri, buf and bufpp are all nao*nao
     //buf=mo_coeff*buf_eri; //nao**2=(nao**2)@(nao**2)
     //bufpp=buf*mo_coeff; //nao**2=(nao**2)@(nao**2)
     pm->dev_push(d_buf_eri, buf_eri, _size_buf_eri * sizeof(double));
     // buf=eri*mo_coeff (nao*nao)@(nao*nao)
     cublasDsymm(dd->handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, nao, nao,
              &alpha, d_buf_eri, nao, d_mo_coeff , nao,
              &beta, d_buf, nao);//i_start is 0, omitting i_start * nao term in mo_coeff 
     if (i==0){
     pm->dev_pull(d_buf, h_buf, _size_buf * sizeof(double));
     for(int x1=0;x1<nao;++x1){for(int x2=0;x2<=x1;++x2){printf("%f  ",buf_eri[x1 * nao + x2]);}printf("\n");}
     for(int x1=0;x1<nao;++x1){for(int x2=0;x2<=nao;++x2){printf("%f  ",h_buf[x1 * nao + x2]);}printf("\n");}
     }
     
     _CUDA_CHECK_ERRORS();
     // buf=mo_coeff*buf (nao*nao)@(nao*nao)
     cublasDgemm(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, nao,
              &alpha, d_mo_coeff , nao, d_buf, nao,
              &beta, d_buf_bufpp, nao); // j_start is 0, omitting j_start * nao term in mo_coeff
     _CUDA_CHECK_ERRORS();
     pm->dev_pull(d_buf_bufpp, bufpp + i * nao * nao, _size_buf_bufpp * sizeof(double));
     }
  pm->dev_free(d_mo_coeff);
  pm->dev_free(d_buf);
  pm->dev_free(d_buf_bufpp);
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
   //return 0;
  }
#endif //for fastest
//#ifdef _SIMPLE_TIMER
//    double t1 = omp_get_wtime();
//    t_array[5] += t1 - t0;
//#endif
}
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
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::hessop_push_bPpj()\n");
#endif

  profile_start("hessop_push_bPpj");
  
  py::buffer_info info_bPpj = _bPpj.request(); // 3D array (naux, nmo, nocc) : read-only

  double * bPpj = static_cast<double*>(info_bPpj.ptr);

  const int device_id = 0;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);
  
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

  pm->dev_push_async(d_bPpj, bPpj, _size_bPpj*sizeof(double), dd->stream);

  profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::hessop_push_bPpj()\n");
#endif
}

/* ---------------------------------------------------------------------- */

void Device::hessop_get_veff(int naux, int nmo, int ncore, int nocc,
		    py::array_t<double> _bPpj, py::array_t<double> _vPpj, py::array_t<double> _vk_bj)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::hessop_get_veff()\n");
#endif
  
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  profile_start("hessop_get_veff");
  
  py::buffer_info info_vPpj = _vPpj.request(); // 3D array (naux, nmo, nocc) : read-only 
  py::buffer_info info_vk_bj = _vk_bj.request(); // 2D array (nmo-ncore, nocc) : accumulate
  
  double * vPpj = static_cast<double*>(info_vPpj.ptr);
  double * vk_bj = static_cast<double*>(info_vk_bj.ptr);

  const int device_id = 0;
  
  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);
  
  int nvirt = nmo - ncore;

#if 0
  py::buffer_info info_bPpj = _bPpj.request(); // 3D array (naux, nmo, nocc) : read-only
  double * bPpj = static_cast<double*>(info_bPpj.ptr);
  
  printf("LIBGPU:: naux= %i  nmo= %i  ncore= %i  nocc= %i  nvirt= %i  dd->size_buf= %i\n",naux, nmo, ncore, nocc, nvirt,dd->size_buf);
  printf("LIBGPU:: shape : bPpj= (%i, %i, %i)  vPj= (%i, %i, %i)  vk_bj= (%i, %i)\n",
	 info_bPpj.shape[0],info_bPpj.shape[1],info_bPpj.shape[2],
	 info_vPpj.shape[0],info_vPpj.shape[1],info_vPpj.shape[2],
	 info_vk_bj.shape[0], info_vk_bj.shape[1]);
#endif

  // this buf realloc needs to be consistent with that in init_get_jk()
  
  int _size_buf = naux * nmo * nocc;
  if(_size_buf > dd->size_buf) {
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: updating buffer sizes\n");
#endif
    dd->size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double*) pm->dev_malloc_host(2*_size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(_size_buf*sizeof(double));
    buf4 = (double *) pm->dev_malloc_host(_size_buf*sizeof(double));

    if(dd->d_buf1) pm->dev_free(dd->d_buf1);
    if(dd->d_buf2) pm->dev_free(dd->d_buf2);
    if(dd->d_buf3) pm->dev_free(dd->d_buf3);
    
    dd->d_buf1 = (double *) pm->dev_malloc(_size_buf * sizeof(double));
    dd->d_buf2 = (double *) pm->dev_malloc(_size_buf * sizeof(double));
    dd->d_buf3 = (double *) pm->dev_malloc(_size_buf * sizeof(double));
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
  //  pm->dev_barrier();
  pm->dev_push_async(dd->d_buf1, vPpj, naux*nmo*nocc*sizeof(double), dd->stream);

  {
    dim3 grid_size(nvirt, nocc, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
    
    _hessop_get_veff_reshape1<<<grid_size, block_size, 0, dd->stream>>>(d_vPpj, dd->d_buf1, nmo, nocc, ncore, nvirt, naux);
  }
  
  {
    dim3 grid_size(nocc, naux, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
  
    _hessop_get_veff_reshape2<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf2, d_bPpj, nmo, nocc, ncore, nvirt, naux);
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
    
    cublasDgemm(dd->handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dd->d_buf2, lda, d_vPpj, ldb, &beta, d_vk_bj, ldc);
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
  
    _hessop_get_veff_reshape3<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf2, d_bPpj, nmo, nocc, ncore, nvirt, naux);
  }
  
  {
    dim3 grid_size(naux, ncore, 1);
    dim3 block_size(1, 1, _HESSOP_BLOCK_SIZE);
  
    _hessop_get_veff_reshape4<<<grid_size, block_size, 0, dd->stream>>>(d_vPpj, dd->d_buf1, nmo, nocc, ncore, nvirt, naux);
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
    
    cublasDgemm(dd->handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_vPpj, lda, dd->d_buf2, ldb, &beta, d_vk_bj, ldc);
  }

  pm->dev_pull_async(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double), dd->stream);
  pm->dev_stream_wait(dd->stream);
    
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
  
  // pm->dev_stream_wait(dd->stream);
  // pm->dev_pull(d_vk_bj, vk_bj, _size_vk_bj*sizeof(double));
  
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

  profile_stop();
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[3] += t1 - t0;
#endif
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Leaving Device::hessop_get_veff()\n");
#endif
  
}

#endif
