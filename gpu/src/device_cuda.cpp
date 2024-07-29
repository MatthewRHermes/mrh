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

//#define _DEBUG_DEVICE
//#define _DEBUG_H2EFF
//#define _DEBUG_CONTRACT1
//#define _DEBUG_CONTRACT2
//#define _DEBUG_H2EFF2
#define _DEBUG_H2EFF_DF
#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

int * Device::dd_fetch_pumap(my_device_data * dd, int size_pumap_, int type_pumap = _PUMAP_2D_UNPACK)
{
  // search if pack/unpack map already created

  int indx = -1;
  for(int i=0; i<dd->size_pumap.size(); ++i)
    if(dd->type_pumap[i] == type_pumap && dd->size_pumap[i] == size_pumap_) indx = i;

  // add unpack/pack map if not found
  
  if(indx < 0) {
    dd->type_pumap.push_back(type_pumap);
    dd->size_pumap.push_back(size_pumap_);
    dd->pumap.push_back(nullptr);
    dd->d_pumap.push_back(nullptr);

    indx = dd->type_pumap.size() - 1;

    int size_pumap = -1;
    
    if(type_pumap == _PUMAP_2D_UNPACK) {
      int nao = size_pumap_;
      size_pumap = nao * nao;
      
      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc(size_pumap * sizeof(int));
      
      int _i, _j, _ij;
      int * tm = dd->pumap[indx];
      for(_ij = 0, _i = 0; _i < nao; _i++)
	for(_j = 0; _j<=_i; _j++, _ij++) {
	  tm[_i*nao + _j] = _ij;
	  tm[_i + nao*_j] = _ij;
	}
      
    } else if(type_pumap == _PUMAP_H2EFF_UNPACK) {

      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas * ncas * ncas;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      for (int _i=0; _i<ncas;++_i){
	for (int _j=0, _jk=0; _j<ncas; ++_j){
	  for (int _k=0;_k<=_j;++_k,++_jk){
	    tm[_i*ncas*ncas + _j*ncas+_k]=_i*ncas_pair+_jk;
	    tm[_i*ncas*ncas + _k*ncas+_j]=_i*ncas_pair+_jk;
	  }
	}
      }

    } else if(type_pumap == _PUMAP_H2EFF_PACK) {

      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas * ncas_pair;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      int _i, _j, _k, _ijk;
      for (_ijk=0, _i=0; _i<ncas;++_i){
	for (_j=0; _j<ncas; ++_j){
	  for (_k=0;_k<=_j;++_k,++_ijk){
	    tm[_ijk] = _i*ncas*ncas + _j*ncas+_k;
	  }
	}
      }
      
    } // if(type_pumap)
    
    pm->dev_push_async(dd->d_pumap[indx], dd->pumap[indx], size_pumap*sizeof(int), dd->stream);
  } // if(map_not_found)
  
  // set device pointer to current map
  
  dd->d_pumap_ptr = dd->d_pumap[indx];

  return dd->d_pumap_ptr;
}

/* ---------------------------------------------------------------------- */
void Device::transfer_mo_coeff(py::array_t<double> _mo_coeff, int _size_mo_coeff)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  py::buffer_info info_mo_coeff = _mo_coeff.request(); // 2D array (naux, nao_pair)
  double * mo_coeff = static_cast<double*>(info_mo_coeff.ptr);
  const int device_id = 0;//count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  push_mo_coeff(dd, mo_coeff, _size_mo_coeff);
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
#endif
}
/* ---------------------------------------------------------------------- */
void Device::push_mo_coeff(my_device_data * dd, double * mo_coeff, int _size_mo_coeff)
{
if (_size_mo_coeff > dd->size_mo_coeff){
    dd->size_mo_coeff = _size_mo_coeff;
    if (dd->d_mo_coeff) pm->dev_free(dd->d_mo_coeff);
    dd->d_mo_coeff = (double *) pm->dev_malloc(_size_mo_coeff*sizeof(double));
  } 
  pm->dev_push(dd->d_mo_coeff,mo_coeff,_size_mo_coeff*sizeof(double));
  //pm->dev_push_async(dd->d_mo_coeff,mo_coeff,_size_mo_coeff*sizeof(double));
}
/* ---------------------------------------------------------------------- */

double * Device::dd_fetch_eri(my_device_data * dd, double * eri1, size_t addr_dfobj, int count)
{
#if defined(_DEBUG_DEVICE) || defined(_DEBUG_ERI_CACHE)
  return dd_fetch_eri_debug(dd, eri1, addr_dfobj, count);
#endif

  double * d_eri;
  
  // retrieve id of cached eri block
  
  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count) {
      id = i;
      break;
    }
  
  // grab/update cached data
  
  if(id < eri_list.size()) {
    
    eri_count[id]++;
    d_eri = d_eri_cache[id];
    
    if(update_dfobj) {
      eri_update[id]++;
      int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), dd->stream);
      if(err) {
	printf("LIBGPU:: dev_push_async(d_eri) updating eri block\n");
	exit(1);
      }
    }
    
  } else {
    
    eri_list.push_back(addr_dfobj+count);
    eri_count.push_back(1);
    eri_update.push_back(0);
    eri_size.push_back(naux * nao_pair);
    eri_device.push_back(dd->device_id);
    
    eri_num_blocks.push_back(0); // grow array
    eri_num_blocks[id-count]++;  // increment # of blocks for this dfobj
    
    eri_extra.push_back(naux);
    eri_extra.push_back(nao_pair);
    
    int id = d_eri_cache.size();
    
    d_eri_cache.push_back( (double *) pm->dev_malloc(naux * nao_pair * sizeof(double)) );
    d_eri = d_eri_cache[ id ];
    
    int err = pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double), dd->stream);
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
      exit(1);
    }
    
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */

double * Device::dd_fetch_eri_debug(my_device_data * dd, double * eri1, size_t addr_dfobj, int count)
{   
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting eri_cache lookup\n");
#endif

  double * d_eri;
  
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
    eri_device.push_back(dd->device_id);
    
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

  return d_eri;
}

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

  if(!use_eri_cache) {
    int _size_eri1 = naux * nao_pair;
    if(_size_eri1 > dd->size_eri1) {
      dd->size_eri1 = _size_eri1;
      if(dd->d_eri1) pm->dev_free(dd->d_eri1);
      dd->d_eri1 = (double *) pm->dev_malloc(_size_eri1 * sizeof(double));
    }
  }
  
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
  
  dd_fetch_pumap(dd, nao);
  
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
  
  DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair, pm, DA_HOST);
  //  printf("LIBGPU:: eri1= %p  dfobj= %lu  count= %i  combined= %lu\n",eri1,addr_dfobj,count,addr_dfobj+count);
  printf("LIBGPU:: dfobj= %#012x  count= %i  combined= %#012x  update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count, update_dfobj);
  printf("LIBGPU::     0:      %f %f %f %f\n",da_eri1(0,0), da_eri1(0,1), da_eri1(0,nao_pair-2), da_eri1(0,nao_pair-1));
  printf("LIBGPU::     1:      %f %f %f %f\n",da_eri1(1,0), da_eri1(1,1), da_eri1(1,nao_pair-2), da_eri1(1,nao_pair-1));
  printf("LIBGPU::     naux-2: %f %f %f %f\n",da_eri1(naux-2,0), da_eri1(naux-2,1), da_eri1(naux-2,nao_pair-2), da_eri1(naux-2,nao_pair-1));
  printf("LIBGPU::     naux-1: %f %f %f %f\n",da_eri1(naux-1,0), da_eri1(naux-1,1), da_eri1(naux-1,nao_pair-2), da_eri1(naux-1,nao_pair-1));
#endif
  
  if(use_eri_cache)
    d_eri = dd_fetch_eri(dd, eri1, addr_dfobj, count);

  profile_stop();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting with_j calculation\n");
#endif
    if (with_j){
	 
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
    
    _getjk_unpack_buf2<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf2, d_eri, dd->d_pumap_ptr, naux, nao, nao_pair);
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
  const int device_id = 0;//count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::df_ao2mo_pass1_fdrv()\n");
#endif
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  profile_start(" df_ao2mo_pass1_fdrv setup\n");
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  py::buffer_info info_bufpp = _bufpp.request(); // 3D array (naux,nmo,nmo)
#ifdef _DEBUG_DEVICE
  printf("LIBGPU::: naux= %i  nmo= %i  nao= %i  blksize=%i \n",naux,nmo,nao,blksize);
  printf("LIBGPU::shape: _eri1= (%i,%i)  _mo_coeff= (%i,%i)  _bufpp= (%i, %i, %i)\n",
  	 info_eri1.shape[0], info_eri1.shape[1],
  	 info_mo_coeff.shape[0], info_mo_coeff.shape[1],
  	 info_bufpp.shape[0], info_bufpp.shape[1],info_bufpp.shape[2]);
#endif
  const int nao_pair = nao*(nao+1)/2;
  double * eri = static_cast<double*>(info_eri1.ptr);
  int _size_eri = naux * nao_pair;
  int _size_eri_unpacked = naux * nao * nao; 
  double * bufpp = static_cast<double*>(info_bufpp.ptr);
  //int _size_bufpp = naux * nao * nao;
#if 1
  double * d_mo_coeff = dd->d_mo_coeff;
#else
  py::buffer_info info_mo_coeff = _mo_coeff.request(); // 2D array (nmo, nmo)
  double *d_mo_coeff= (double*) pm->dev_malloc(_size_mo_coeff*sizeof(double));//allocate mo_coeff (might be able to avoid if already used in get_jk)
  pm->dev_push(d_mo_coeff, mo_coeff, _size_mo_coeff * sizeof(double));//doing this allocation and pushing first because it doesn't change over iteration 
  double * mo_coeff = static_cast<double*>(info_mo_coeff.ptr);
  int _size_mo_coeff = nao * nao;
#endif
#ifdef _DEBUG_DEVICE
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
  {
  double * d_buf_1 = (double*) pm->dev_malloc(_size_eri_unpacked*sizeof(double));//new
  double * d_buf_2 = (double*) pm->dev_malloc(_size_eri_unpacked*sizeof(double));//new
  double * d_buf = d_buf_1; //for eri*mo_coeff (don't pull or push) 
  double * d_eri_unpacked = d_buf_2;//set memory for the entire eri array on GPU
  //unpack 2D eri of size naux * nao(nao+1)/2 to a full naux*nao*nao 3D matrix
  double * d_eri = (double*) pm->dev_malloc (sizeof(double)* _size_eri);//set memory for the entire eri array on GPU
  pm->dev_push(d_eri, eri, _size_eri * sizeof(double));//doing this allocation and pushing first because it doesn't change over iterations. 
  int _size_tril_map = nao * nao;
  int * my_tril_map = (int*) malloc (_size_tril_map * sizeof(int));
  int * my_d_tril_map = (int*) pm->dev_malloc (_size_tril_map * sizeof(int));
    int _i, _j, _ij;
    for(_ij = 0, _i = 0; _i < nao; ++_i)
      for(_j = 0; _j<=_i; ++_j, ++_ij) {
    	my_tril_map[_i*nao + _j] = _ij;
    	my_tril_map[_i + nao*_j] = _ij;
    } 
  int * my_d_tril_map_ptr=my_d_tril_map;
  pm->dev_push(my_d_tril_map,my_tril_map,_size_tril_map*sizeof(int));
  dim3 grid_size(_TILE(naux,_UNPACK_BLOCK_SIZE), _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
  dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  _getjk_unpack_buf2<<<grid_size,block_size,0,dd->stream>>>(d_eri_unpacked, d_eri, my_d_tril_map_ptr, naux, nao, nao_pair);
#ifdef _DEBUG_DEVICE
  printf("Finished unpacking\n");
#endif
  double alpha = 1.0;
  double beta = 0.0;
  //bufpp = mo.T @ eri @ mo 
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- calling cublasDgemmStrideBatched() in ao2mo_fdrv\n");
#endif
  profile_stop ();
  profile_start(" df_ao2mo_pass1_fdrv StridedBatchedDgemm\n");
  //buf = np.einsum('ijk,kl->ijl',eri_unpacked,mo_coeff),i=naux,j=nao,l=nao 
  cublasDgemmStridedBatched(dd->handle, 
                      CUBLAS_OP_N, CUBLAS_OP_N,nao, nao, nao, 
                      &alpha,d_eri_unpacked, nao, nao*nao,d_mo_coeff, nao, 0, 
                      &beta,d_buf, nao, nao*nao,naux);
  _CUDA_CHECK_ERRORS();
  //pm->dev_free(d_eri_unpacked);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- calling cublasDgemmStrideBatched() in ao2mo_fdrv \n");
#endif
  //bufpp = np.einsum('jk,ikl->ijl',mo_coeff.T,buf),i=naux,j=nao,l=nao 
  //double * d_bufpp = (double*) pm->dev_malloc (sizeof(double)* _size_bufpp);//set memory for the entire bufpp array, no pushing needed 
  double * d_bufpp = d_buf_2;//set memory for the entire bufpp array, no pushing needed 
  cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, nao, 
                      &alpha,d_mo_coeff, nao, 0, d_buf, nao, nao*nao, 
                      &beta, d_bufpp, nao, nao*nao,naux);
  _CUDA_CHECK_ERRORS();
  profile_stop();
  profile_start(" df_ao2mo_pass1_fdrv Data pull\n");
  pm->dev_pull(d_bufpp, bufpp, _size_eri_unpacked * sizeof(double));
  pm->dev_free(my_d_tril_map);
  pm->dev_free(d_eri);
  pm->dev_free(d_buf_1);
  pm->dev_free(d_buf_2);
  profile_stop();
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
  }
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array[5] += t1 - t0;
#endif
}

/* ---------------------------------------------------------------------- */

__global__ void extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo) {
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncas && j < ncas) {
        small_mat[i * ncas + j] = big_mat[(i + ncore) * nmo + (j + ncore)];
    }
}

/* ---------------------------------------------------------------------- */

__global__ void transpose_2310(double * in, double * out, int nmo, int ncas) {
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

__global__ void transpose_3210(double* in, double* out, int nmo, int ncas) {
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

  if(i >= nmo*ncas) return;
  if(j >= ncas*ncas) return;

  double * out_buf = &(out[i * ncas_pair]);
  double * in_buf = &(in[i * ncas*ncas]);

  out_buf[map[j]] = in_buf[ j ];
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

void Device::update_h2eff_sub(int ncore, int ncas, int nocc, int nmo,
                              py::array_t<double> _umat, py::array_t<double> _h2eff_sub)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting update_h2eff_sub function\n");
#endif
  
  profile_start("Setup initial h2eff_sub");
  
  py::buffer_info info_umat = _umat.request(); // 2d array nmo*nmo
  py::buffer_info info_h2eff_sub = _h2eff_sub.request();// 2d array nmo * (ncas*(ncas*(ncas+1)/2))

  const int device_id = 0;//count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  const int ncas_pair = ncas * (ncas+1)/2;

  double * umat = static_cast<double*>(info_umat.ptr);
  double * h2eff_sub = static_cast<double*>(info_h2eff_sub.ptr);

#ifdef _DEBUG_DEVICE
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting h2eff_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
  
  int _size_h2eff_unpacked=nmo*ncas*ncas*ncas;
  int _size_h2eff_packed=nmo*ncas*ncas_pair;

  if(_size_h2eff_unpacked > dd->size_buf) {
    dd->size_buf = _size_h2eff_unpacked;

    if(dd->d_buf1) pm->dev_free_async(dd->d_buf1, dd->stream);
    if(dd->d_buf2) pm->dev_free_async(dd->d_buf2, dd->stream);

    dd->d_buf1 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double), dd->stream);
    dd->d_buf2 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double), dd->stream);
  }

  double * d_h2eff_unpacked = dd->d_buf1;

  if(ncas*ncas > dd->size_ucas) {
    dd->size_ucas = ncas * ncas;
    if(dd->d_ucas) pm->dev_free_async(dd->d_ucas, dd->stream);
    dd->d_ucas = (double *) pm->dev_malloc_async(dd->size_ucas * sizeof(double), dd->stream);
  }
  
  if(nmo*nmo > dd->size_umat) {
    dd->size_umat = nmo * nmo;
    if(dd->d_umat) pm->dev_free_async(dd->d_umat, dd->stream);
    dd->d_umat = (double *) pm->dev_malloc_async(dd->size_umat * sizeof(double), dd->stream);
  }

  pm->dev_push_async(dd->d_umat, umat, nmo*nmo*sizeof(double), dd->stream);
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Setup update function\n");
#endif
  
  profile_next("extraction");
  
  //ucas = umat[ncore:nocc, ncore:nocc]
  {
    dim3 blockDim(_UNPACK_BLOCK_SIZE);
    dim3 gridDim(_TILE(ncas,blockDim.x), _TILE(ncas,blockDim.y));
    extract_submatrix<<<gridDim, blockDim, 0, dd->stream>>>(dd->d_umat, dd->d_ucas, ncas, ncore, nmo);
  }
  
  //h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2) Initially h2eff_sub is nmo*(ncas*ncas_pair)
  //h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)

  if(_size_h2eff_packed > dd->size_h2eff) {
    dd->size_h2eff = _size_h2eff_packed;
    if(dd->d_h2eff) pm->dev_free_async(dd->d_h2eff, dd->stream);
    dd->d_h2eff = (double *) pm->dev_malloc_async(dd->size_h2eff * sizeof(double), dd->stream);
  }
  
  double * d_h2eff_sub = dd->d_h2eff;
  
  pm->dev_push_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed*sizeof(double), dd->stream);

  profile_next("map creation and pushed");
  
  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_UNPACK);

  profile_next("unpacking");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- created and pushed unpacking map\n");
#endif

  {
    dim3 blockDim(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    dim3 gridDim(_TILE(nmo,_UNPACK_BLOCK_SIZE), _TILE(ncas*ncas*ncas,_UNPACK_BLOCK_SIZE)); 
    _unpack_h2eff_2d<<<gridDim, blockDim, 0, dd->stream>>>(d_h2eff_sub, d_h2eff_unpacked, d_my_unpack_map_ptr, nmo, ncas,ncas_pair);
    _CUDA_CHECK_ERRORS();
  }
  
  profile_next("2 dgemms");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- unpacked h2eff_sub \n");
#endif

  //1. h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))) # bpaa
  //2. h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(1))) # qbaa
  //3. h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbab
  //4. h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbbb
  // doing 3,4,tranpose, 1,2, tranpose

  const double alpha=1.0;
  const double beta=0.0;

  //h2eff_step1=([pi]jk,jJ->[pi]kJ)

  double * d_h2eff_step1 = dd->d_buf2;

  cublasDgemmStridedBatched(dd->handle,CUBLAS_OP_N,CUBLAS_OP_N,ncas,ncas,ncas,
			    &alpha, 
			    dd->d_ucas, ncas, 0,
			    d_h2eff_unpacked, ncas,ncas*ncas, 
			    &beta, d_h2eff_step1, ncas, ncas*ncas, ncas*nmo);
  _CUDA_CHECK_ERRORS();

  //h2eff_step2=([pi]kJ,kK->[pi]JK
  
  double * d_h2eff_step2 = dd->d_buf1;
  
  cublasDgemmStridedBatched(dd->handle,CUBLAS_OP_N,CUBLAS_OP_T,ncas,ncas,ncas,
			    &alpha, 
			    d_h2eff_step1, ncas,ncas*ncas, 
			    dd->d_ucas, ncas, 0,
			    &beta, d_h2eff_step2, ncas, ncas*ncas, ncas*nmo);
  _CUDA_CHECK_ERRORS();

  profile_next("transpose");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished first 2 cublasDgemmStridedBatched Functions \n");
#endif
  
  double * d_h2eff_transposed = dd->d_buf2;

  //h2eff_tranposed=(piJK->JKip)
  {
    dim3 blockDim(1,1,_DEFAULT_BLOCK_SIZE);
    dim3 gridDim(_TILE(nmo,blockDim.x),_TILE(ncas,blockDim.y),_TILE(ncas,blockDim.z));
    transpose_2310<<<gridDim, blockDim, 0, dd->stream>>>(d_h2eff_step2, d_h2eff_transposed, nmo,ncas);
  }
 
  
  profile_next("last 2 dgemm");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished transposing\n");
#endif
  
  double * d_h2eff_step3 = dd->d_buf1;
  
  //h2eff_sub=np.einsum('iI,JKip->JKIp',ucas,h2eff_sub) h2eff=ncas,ncas,ncas,nmo; ucas=ncas,ncas

  cublasDgemmStridedBatched(dd->handle,CUBLAS_OP_N,CUBLAS_OP_T,nmo,ncas,ncas,
			    &alpha, d_h2eff_transposed, nmo, ncas*nmo, dd->d_ucas, ncas, 0, 
			    &beta, d_h2eff_step3, nmo, ncas*nmo, ncas*ncas);
  _CUDA_CHECK_ERRORS();

  //h2eff_step4=([JK]Ip,pP->[JK]IP)

  double * d_h2eff_step4 = dd->d_buf2;

  cublasDgemmStridedBatched(dd->handle,CUBLAS_OP_N,CUBLAS_OP_N,nmo,ncas,nmo,
	&alpha, dd->d_umat, nmo, 0, d_h2eff_step3, nmo, ncas*nmo, 
	&beta, d_h2eff_step4, nmo, ncas*nmo, ncas*ncas);
  
  profile_next("2nd transpose");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished last 2 cublasDgemmStridedBatched Functions \n");
#endif
  
  //h2eff_tranposed=(JKIP->PIJK) 3201

  double * d_h2eff_transpose2 = dd->d_buf1;
  {
    dim3 blockDim(1,1,_DEFAULT_BLOCK_SIZE);
    dim3 gridDim(_TILE(ncas,blockDim.x),_TILE(ncas,blockDim.y),_TILE(ncas,blockDim.z));
    transpose_3210<<<gridDim, blockDim, 0, dd->stream>>>(d_h2eff_step4, d_h2eff_transpose2, nmo, ncas);
  }
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- finished transposing back\n");
#endif
  
  //ix_i, ix_j = np.tril_indices (ncas)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas*ncas)
  //h2eff_sub = h2eff_sub[:,:,(ix_i*ncas)+ix_j]
  //h2eff_sub = h2eff_sub.reshape (nmo, -1)

  profile_next("second map and packing");
  
  int * d_my_pack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_PACK);

  {
    dim3 blockDim(1, _UNPACK_BLOCK_SIZE, 1);
    dim3 gridDim(nmo, _TILE(ncas*ncas_pair,_UNPACK_BLOCK_SIZE)); 
    _pack_h2eff_2d<<<gridDim, blockDim, 0, dd->stream>>>(d_h2eff_transpose2, d_h2eff_sub, d_my_pack_map_ptr, nmo, ncas,ncas_pair);
  }
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Freed map\n");
#endif
  
  pm->dev_pull_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed*sizeof(double), dd->stream);

  pm->dev_stream_wait(dd->stream);
  
  profile_stop();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: -- Leaving update function\n");
  cudaMemGetInfo(&freeMem, &totalMem);
  
  printf("Ending h2eff_sub_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[6] += t1 - t0;
#endif
}

/* ---------------------------------------------------------------------- */

__global__ void transpose_210(double * in, double * out, int naux, int nao, int ncas) {
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

void Device::h2eff_df_contract1(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int blksize, 
                                py::array_t<double> _mo_cas,py::array_t<double> _bmuP)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting h2eff_df_contract1 function");
#endif 
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  py::buffer_info info_bmuP = _bmuP.request(); //2D array nao * ncas
  const int device_id = 0;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  const int nao_pair = nao * (nao+1)/2;
  const int _size_bmuP = nao*ncas*naux;
  const int _size_cderi = naux*nao_pair;
  const int _size_cderi_unpacked = naux*nao*nao;
  double * bmuP = static_cast<double*>(info_bmuP.ptr);
  double * cderi = static_cast<double*>(info_cderi.ptr);
#if 1
  double * d_mo_cas = dd->d_mo_coeff;
#else
  py::buffer_info info_mo_cas = _mo_cas.request(); //2D array nao * ncas
  double * mo_cas = static_cast<double*>(info_mo_cas.ptr);
  const int _size_mo_cas = nao*ncas;
  double * d_mo_cas = (double*) pm->dev_malloc(_size_mo_cas*sizeof(double));
  pm->dev_push(d_mo_cas,mo_cas,_size_mo_cas*sizeof(double));
#endif
  double * d_cderi=(double *) pm->dev_malloc( _size_cderi * sizeof(double));
  pm->dev_push(d_cderi,cderi,_size_cderi*sizeof(double));
  double * d_cderi_unpacked=(double *) pm->dev_malloc( _size_cderi_unpacked * sizeof(double));
  double * d_bPmu = (double*) pm->dev_malloc( _size_bmuP * sizeof(double));
  double * d_bmuP = (double*) pm->dev_malloc( _size_bmuP * sizeof(double));
  int _size_unpack_map = nao*nao;
  int * my_unpack_map = (int*) malloc(_size_unpack_map*sizeof(int));
  for (int _i = 0, _ij = 0; _i < nao ; ++_i)
    for (int _j = 0; _j <= _i; ++_j, ++_ij){
      my_unpack_map[_i*nao + _j]= _ij;
      my_unpack_map[_j*nao + _i]= _ij;
    }
  int * d_my_unpack_map = (int*) pm->dev_malloc(_size_unpack_map*sizeof(int));
  int * d_my_unpack_map_ptr = d_my_unpack_map;
  pm->dev_push(d_my_unpack_map, my_unpack_map,_size_unpack_map*sizeof(int));  
  {dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(naux,_UNPACK_BLOCK_SIZE), _TILE(nao*nao,_UNPACK_BLOCK_SIZE));
  _getjk_unpack_buf2<<<grid_size,block_size, 0, dd->stream>>>(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);}
  
  //bmuP1 = bPmn.contract1 (mo_cas)
  //bmuP1 = np.einsum('Pmn,nu->Pmu',unpack_tril(bPmn),mo_cas)
  // I am doing (Pmn,nu->Pmu,bpmn_up, mo_cas.T) because python does row major storage and cpp does column major 
  const double alpha=1.0;
  const double beta=0.0;
  cublasDgemmStridedBatched (dd->handle,CUBLAS_OP_N, CUBLAS_OP_T, 
                              nao, ncas, nao,
                              &alpha, d_cderi_unpacked, nao, nao*nao, d_mo_cas, ncas, 0,
                              &beta, d_bPmu, nao, ncas*nao, naux);
  _CUDA_CHECK_ERRORS();
  {dim3 block_size(_UNPACK_BLOCK_SIZE, 1, _UNPACK_BLOCK_SIZE);
  dim3 grid_size(_TILE(naux,block_size.x), _TILE(ncas,block_size.y), _TILE(nao,block_size.z));
  transpose_210<<<grid_size,block_size, 0, dd->stream>>>(d_bPmu, d_bmuP,naux,nao,ncas);}
  pm->dev_pull(d_bmuP,bmuP,_size_bmuP*sizeof(double));
  //pm->dev_free(d_mo_cas);
  pm->dev_free(d_cderi);
  pm->dev_free(d_cderi_unpacked);
  pm->dev_free(d_bmuP);
  pm->dev_free(d_bPmu);
  pm->dev_free(d_my_unpack_map);
  free(my_unpack_map);
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;
#endif
}

__global__ void get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao) {
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncas && j < nao) {
        small_mat[i * nao + j] = big_mat[(j+ncore)*nao + i];
    }
}
__global__ void transpose_120(double * in, double * out, int naux, int nao, int ncas) {
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

void Device::get_h2eff_df(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri) 
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting h2eff_df_contract1 function");
#endif 
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas
  const int device_id = 0;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri = nmo*ncas*ncas*ncas_pair;
  const int _size_cderi = naux*nao_pair;
  const int _size_cderi_unpacked = naux*nao*nao;
  const int _size_mo_cas = nao*ncas;
  double * eri = static_cast<double*>(info_eri.ptr);
  double * cderi = static_cast<double*>(info_cderi.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  double * d_mo_cas = (double*) pm->dev_malloc(_size_mo_cas*sizeof(double));
  // 
  // write code to extract d_mo_cas from d_mo_coeff
  {dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(ncas, block_size.x), _TILE(nao, block_size.y));
  get_mo_cas<<<grid_size, block_size, 0, dd->stream>>>(d_mo_coeff, d_mo_cas, ncas, ncore, nao);}
  //
#ifdef _DEBUG_H2EFF_DF2
  double * h_mo_cas = (double*) malloc (_size_mo_cas *sizeof(double));
  pm->dev_pull(d_mo_cas, h_mo_cas,_size_mo_cas*sizeof(double));
  for (int i =0; i<ncas; ++i){
    for (int j =0; j<nao; ++j){
      printf("%f \t",h_mo_cas[i*nao+j]);}printf("\n");}
  free(h_mo_cas);
#endif
  // unpacking business that should really just have been done with stored map already and also with the stored eris
  double * d_cderi=(double*) pm->dev_malloc( _size_cderi * sizeof(double));
  pm->dev_push(d_cderi,cderi,_size_cderi*sizeof(double));
  double * d_cderi_unpacked=(double*) pm->dev_malloc( _size_cderi_unpacked * sizeof(double));
  int _size_unpack_map = nao*nao;
  int * my_unpack_map = (int*) malloc(_size_unpack_map*sizeof(int));
  for (int _i = 0, _ij = 0; _i < nao ; ++_i)
    for (int _j = 0; _j <= _i; ++_j, ++_ij){
      my_unpack_map[_i*nao + _j]= _ij;
      my_unpack_map[_j*nao + _i]= _ij;
    }
  int * d_my_unpack_map = (int*) pm->dev_malloc(_size_unpack_map*sizeof(int));
  int * d_my_unpack_map_ptr = d_my_unpack_map;
  pm->dev_push(d_my_unpack_map, my_unpack_map,_size_unpack_map*sizeof(int));  
  {dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
  dim3 grid_size(_TILE(naux,_UNPACK_BLOCK_SIZE), _TILE(nao*nao,_UNPACK_BLOCK_SIZE));
  _getjk_unpack_buf2<<<grid_size,block_size, 0, dd->stream>>>(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);}

  
//bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  const double alpha=1.0;
  const double beta=0.0;
  const int _size_bPmu = naux*ncas*nao;
  double * d_bPmu = (double*) pm->dev_malloc (_size_bPmu *sizeof(double));
  //printf("First DGemm\n"); 
  cublasDgemmStridedBatched (dd->handle,CUBLAS_OP_N, CUBLAS_OP_N, 
                              nao, ncas, nao,
                              &alpha, d_cderi_unpacked, nao, nao*nao, d_mo_cas, nao, 0,
                              &beta, d_bPmu, nao, ncas*nao, naux);
  _CUDA_CHECK_ERRORS();
#ifdef _DEBUG_H2EFF_DF2
  double * h_bPmu = (double*) malloc (_size_bPmu *sizeof(double));
  pm->dev_pull(d_bPmu, h_bPmu,_size_bPmu*sizeof(double));
  for (int i =0; i<ncas; ++i){
    for (int j =0; j<nao; ++j){
      printf("%f \t",h_bPmu[i*nao+j]);}printf("\n");}
  free(h_bPmu);
#endif

//bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)
  //cublasDgemmStridedBatched(batch = P, contracted dimension = m)
  const int _size_bPvu = naux*ncas*ncas;
  double * d_bPvu = (double*) pm->dev_malloc (_size_bPvu *sizeof(double));
  //printf("Second DGemm\n"); 
  cublasDgemmStridedBatched (dd->handle, CUBLAS_OP_C, CUBLAS_OP_N, 
                             ncas, ncas, nao, 
                             &alpha, d_mo_cas, nao, 0,
                             d_bPmu, nao, ncas*nao, 
                             &beta, d_bPvu, ncas, ncas*ncas, naux); 
#ifdef _DEBUG_H2EFF_DF2
  double * h_bPvu = (double*) malloc (_size_bPvu *sizeof(double));
  pm->dev_pull(d_bPvu, h_bPvu,_size_bPvu*sizeof(double));
  for (int h=0; h<4; ++h){
  for (int i =0; i<ncas; ++i){
    for (int j =0; j<ncas; ++j){
      printf("%f \t",h_bPvu[h*ncas*ncas+i*ncas+j]);}printf("\n");}printf("\n");}
  free(h_bPvu);
#endif


//eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)
  //transpose bPmu 
  double * d_bumP = (double*) pm->dev_malloc (_size_bPmu *sizeof(double));
  {dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(naux, block_size.x),_TILE(nao, block_size.y),_TILE(ncas, block_size.z));
  transpose_120<<<grid_size, block_size, 0, dd->stream>>>(d_bPmu, d_bumP, naux, ncas, nao);}
#ifdef _DEBUG_H2EFF_DF2
  printf("printing bPmu -> bumP\n");
  double * h_bumP = (double*) malloc (_size_bPmu *sizeof(double));
  pm->dev_pull(d_bumP, h_bumP,_size_bPmu*sizeof(double));
  for (int i =0; i<1; ++i){
    for (int j =0; j<naux; ++j){
      printf("%f \t",h_bumP[i*naux+j]);}printf("\n");}
  free(h_bumP);
#endif


  //transpose bPvu
  double * d_buvP = (double*) pm->dev_malloc (_size_bPvu *sizeof(double));
  {dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(naux, block_size.x),_TILE(ncas, block_size.y),_TILE(ncas, block_size.z));
  transpose_210<<<grid_size, block_size, 0, dd->stream>>>(d_bPvu, d_buvP, naux, ncas, ncas);}
#ifdef _DEBUG_H2EFF_DF2
  printf("printing bPvu -> bvuP\n");
  double * h_bvuP = (double*) malloc (_size_bPvu *sizeof(double));
  pm->dev_pull(d_buvP, h_bvuP,_size_bPvu*sizeof(double));
  for (int i =0; i<1; ++i){
    for (int j =0; j<naux; ++j){
      printf("%f \t",h_bvuP[i*naux+j]);}printf("\n");}
  free(h_bvuP);
#endif

   //h_vuwm[i*ncas*nao+j]+=h_bvuP[i*naux + k]*h_bumP[j*naux+k];
  //dgemm (probably just simple, not strided/batched, contracted dimension = P)
  const int _size_mwvu = nao*ncas*ncas*ncas;
  double * d_vuwm = (double*) pm ->dev_malloc( _size_mwvu*sizeof(double));
 // cublasDgemm(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N,
 //             ncas*ncas, ncas*nao, naux,
 //             &alpha, 
 //             d_buvP, naux,
 //             d_bumP, naux,
 //             &beta, 
 //             d_vuwm, ncas*ncas);
cublasDgemm(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N,
              ncas*nao, ncas*ncas, naux,
              &alpha, 
              d_bumP, naux,
              d_buvP, naux,
              &beta, 
              d_vuwm, ncas*nao);
#ifdef _DEBUG_H2EFF_DF2
  printf("printing vuwM\n");
  double * h_vuwm = (double*) malloc (_size_mwvu *sizeof(double));
  pm->dev_pull(d_vuwm, h_vuwm,_size_mwvu*sizeof(double));
  for (int i=0; i<ncas; ++i){
  for (int j=0; j<ncas; ++j){
  for (int k=0; k<ncas; ++k){
  for (int l=0; l<nao; ++l){
    printf("%f \t ", h_vuwm[((i*ncas+j)*ncas+k)*nao+l]);}printf("\n");}printf("\n");}printf("\n");}
  free(h_vuwm);
#endif



//eri = np.einsum('mM,mwvu->Mwvu', mo_coeff.conjugate(),eri)
  //cublasDgemmStridedBatched (batch = v*u, contracted dimenion = m)
  double * d_vuwM = (double*) pm ->dev_malloc( _size_mwvu*sizeof(double));
  cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_T, CUBLAS_OP_C, 
                            ncas, nao, nao, 
                            &alpha, 
                            d_vuwm, nao, ncas*nao,
                            d_mo_coeff, nao, 0,
                            &beta, 
                            d_vuwM, ncas, ncas*nao, ncas*ncas);
#ifdef _DEBUG_H2EFF_DF
  //printf("printing vuwM\n");
  double * h_vuwM = (double*) malloc (_size_mwvu *sizeof(double));
  pm->dev_pull(d_vuwM, h_vuwM,_size_mwvu*sizeof(double));
  for (int i=0; i<ncas; ++i){
  for (int j=0; j<ncas; ++j){
  for (int k=0; k<ncas; ++k){
  for (int l=0; l<nao; ++l){
    printf("%f \t ", h_vuwM[((i*ncas+j)*ncas+k)*nao+l]);}printf("\n");}printf("\n");}printf("\n");}
  //free(h_vuwm);
#endif
  int _size_pack_map = ncas*ncas;
  int * my_pack_map = (int*) malloc(_size_pack_map*sizeof(int));
    for (int _i = 0, _ij = 0; _i < ncas ; ++_i)
    for (int _j = 0; _j <= _i; ++_j, ++_ij){
      my_pack_map[_i*ncas + _j]= _ij;
      my_pack_map[_j*ncas + _i]= _ij;
    }
  int * d_my_pack_map = (int*) pm->dev_malloc(_size_pack_map*sizeof(int));
  int * d_my_pack_map_ptr = d_my_pack_map;
  pm->dev_push(d_my_pack_map, my_pack_map,_size_pack_map*sizeof(int));  
  const int _size_eri_packed=nao*ncas*ncas_pair;
  double * d_eri_transposed = (double*) pm->dev_malloc(_size_mwvu*sizeof(double));
  double * d_eri_packed = (double*) pm->dev_malloc(_size_eri_packed*sizeof(double));
  //packing
  {
  //dim3 block_size (1,1,1);
  //dim3 grid_size (_TILE(nao,block_size.x),_TILE(ncas*ncas*ncas,block_size.y),1);
  //_transpose<<<grid_size,block_size, 0, dd->stream>>>(d_eri_transposed,d_vuwM,nao,ncas*ncas*ncas);
  }
  {dim3 block_size(1,1,1);
  dim3 grid_size(_TILE(nao, block_size.x),_TILE(ncas, block_size.y), _TILE(ncas*ncas,block_size.z));
  //pack_Mwuv<<<grid_size,block_size, 0, dd->stream>>>(d_vuwM, d_eri_packed, d_my_pack_map_ptr, nao, ncas, ncas_pair);
  }
  //pm->dev_pull(d_eri_packed, eri, _size_eri_packed*sizeof(double));
  //pm->dev_free(d_mo_cas);
  for (int i =0, ij=0; i<ncas; ++i){
    for (int j =0; j<=i; ++j,++ij){
      for (int k =0; k<ncas; ++k){
        for (int l =0; l<nao; ++l){
          eri[k*ncas_pair*nao+l*ncas_pair+my_pack_map[i*ncas+j]]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
          //eri[i*ncas_pair*ncas+j*ncas_pair+kl]=h_vuwM[i*ncas*ncas*ncas+j*ncas*ncas+k*ncas+l];}}}}
  pm->dev_free(d_cderi);
  pm->dev_free(d_cderi_unpacked);
  pm->dev_free(d_bPmu);
  pm->dev_free(d_my_unpack_map);
  free(my_unpack_map);
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  //t_array[8] += t1 - t0;//TODO: add the array size
#endif
}
#endif
