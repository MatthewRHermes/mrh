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
//#define _DEBUG_H2EFF2
//#define _DEBUG_H2EFF_DF
#define _DEBUG_AO2MO
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

#if 1
      int ncas = size_pumap_;
      size_pumap = ncas * ncas;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      int _ij, _i, _j;
      for(_ij = 0, _i = 0; _i < ncas; _i++)
	for(_j = 0; _j<=_i; _j++, _ij++) {
	  tm[_i*ncas + _j] = _ij;
	  tm[_i + ncas*_j] = _ij;
	}
#else
      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas * ncas * ncas;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      for (int _i=0; _i<ncas;++_i) {
	for (int _j=0, _jk=0; _j<ncas; ++_j) {
	  for (int _k=0;_k<=_j;++_k,++_jk) {
	    tm[_i*ncas*ncas + _j*ncas+_k]=_i*ncas_pair+_jk;
	    tm[_i*ncas*ncas + _k*ncas+_j]=_i*ncas_pair+_jk;
	  }
	}
      }
#endif
    } else if(type_pumap == _PUMAP_H2EFF_PACK) {
#if 1
      int ncas = size_pumap_;
      int ncas_pair = ncas * (ncas+1)/2;
      size_pumap = ncas_pair;

      dd->pumap[indx] = (int *) pm->dev_malloc_host(size_pumap * sizeof(int));
      dd->d_pumap[indx] = (int *) pm->dev_malloc(size_pumap * sizeof(int));

      int * tm = dd->pumap[indx];
      int _i, _j, _ij;
      for (_i=0, _ij=0; _i<ncas; ++_i) {
	for (_j=0; _j<=_i; ++_j, ++_ij) {
	  tm[_ij] = _i*ncas + _j;
	}
      }
#else
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
#endif
      
    } // if(type_pumap)
    
    pm->dev_push_async(dd->d_pumap[indx], dd->pumap[indx], size_pumap*sizeof(int), dd->stream);
  } // if(map_not_found)
  
  // set pointers to current map

  dd->pumap_ptr = dd->pumap[indx];
  dd->d_pumap_ptr = dd->d_pumap[indx];

  return dd->d_pumap_ptr;
}

/* ---------------------------------------------------------------------- */

void Device::push_mo_coeff(py::array_t<double> _mo_coeff, int _size_mo_coeff)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  py::buffer_info info_mo_coeff = _mo_coeff.request(); // 2D array (naux, nao_pair)

  double * mo_coeff = static_cast<double*>(info_mo_coeff.ptr);

  // host pushes to each device; optimize later host->device0 plus device-device transfers (i.e. bcast)
  
  for(int id=0; id<num_devices; ++id) {
    
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);
  
    if (_size_mo_coeff > dd->size_mo_coeff){
      dd->size_mo_coeff = _size_mo_coeff;
      if (dd->d_mo_coeff) pm->dev_free_async(dd->d_mo_coeff, dd->stream);
      dd->d_mo_coeff = (double *) pm->dev_malloc_async(_size_mo_coeff*sizeof(double), dd->stream);
    }
    
    pm->dev_push_async(dd->d_mo_coeff, mo_coeff, _size_mo_coeff*sizeof(double), dd->stream);
  }
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;
#endif
}

/* ---------------------------------------------------------------------- */
void Device::init_jk_ao2mo(int ncore, int nmo)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  // host initializes on each device 
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    my_device_data * dd = &(device_data[id]);
    int size_j_pc = ncore*nmo;
    int size_k_pc = ncore*nmo;
    if (size_j_pc > dd->size_j_pc){
      dd->size_j_pc = size_j_pc;
      if (dd->d_j_pc) pm->dev_free_async(dd->d_j_pc, dd->stream);
      dd->d_j_pc = (double *) pm->dev_malloc_async(size_j_pc*sizeof(double), dd->stream);
    }

    if (size_k_pc > dd->size_k_pc){
      dd->size_k_pc = size_k_pc;
      if (dd->d_k_pc) pm->dev_free_async(dd->d_k_pc, dd->stream);
      dd->d_k_pc = (double *) pm->dev_malloc_async(size_k_pc*sizeof(double), dd->stream);
    }
  }
  int _size_buf_j_pc = num_devices*nmo*ncore;
  if(_size_buf_j_pc > size_buf_j_pc) {
    size_buf_j_pc = _size_buf_j_pc;
    if(buf_j_pc) pm->dev_free_host(buf_j_pc);
    buf_j_pc = (double *) pm->dev_malloc_host(_size_buf_j_pc*sizeof(double));
    }
  int _size_buf_k_pc = num_devices*nmo*ncore;
  if(_size_buf_k_pc > size_buf_k_pc) {
    size_buf_k_pc = _size_buf_k_pc;
    if(buf_k_pc) pm->dev_free_host(buf_k_pc);
    buf_k_pc = (double *) pm->dev_malloc_host(_size_buf_k_pc*sizeof(double));
    }
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
#endif
}
/* ---------------------------------------------------------------------- */
void Device::init_ints_ao2mo(int naoaux, int nmo, int ncas)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
    int _size_fxpp = naoaux*nmo*nmo;
    if (_size_fxpp > size_fxpp){
        size_fxpp = _size_fxpp;
        if (pin_fxpp) pm->dev_free_host(pin_fxpp);
        pin_fxpp = (double *) pm->dev_malloc_host(_size_fxpp*sizeof(double));
    }
    int _size_bufpa = naoaux*nmo*ncas;
    if (_size_bufpa > size_bufpa){
        size_bufpa = _size_bufpa;
        if (pin_bufpa) pm->dev_free_host(pin_bufpa);
        pin_bufpa = (double *) pm->dev_malloc_host(_size_bufpa*sizeof(double));
    }
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
#endif
}
/* ---------------------------------------------------------------------- */
void Device::pull_jk_ao2mo(py::array_t<double> _j_pc, py::array_t<double> _k_pc, int nmo, int ncore)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  py::buffer_info info_j_pc = _j_pc.request(); //2D array (nmo*ncore)
  double * j_pc = static_cast<double*>(info_j_pc.ptr);
  double * tmp;
  
  py::buffer_info info_k_pc = _k_pc.request(); //2D array (nmo*ncore)
  double * k_pc = static_cast<double*>(info_k_pc.ptr);
  int size = nmo*ncore;//*sizeof(double);
  // Pulling j_pc from all devices
  for (int i=0; i<num_devices; ++i){
  pm->dev_set_device(i);
  my_device_data * dd = &(device_data[i]);

  if (i==0) tmp = j_pc;
  else tmp = &(buf_j_pc[i*nmo*ncore]);
  
  if (dd->d_j_pc) pm->dev_pull_async(dd->d_j_pc, tmp, size*sizeof(double), dd->stream); 
  }
  // Adding j_pc from all devices
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pm->dev_stream_wait(dd->stream);

    if(i > 0 && dd->d_j_pc) {
      
      tmp = &(buf_j_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) j_pc[j] += tmp[j];
    }
  }
  // Pulling k_pc from all devices
  for (int i=0; i<num_devices; ++i){
  pm->dev_set_device(i);
  my_device_data * dd = &(device_data[i]);

  if (i==0) tmp = k_pc;
  else tmp = &(buf_k_pc[i*nmo*ncore]);

  if (dd->d_k_pc) pm->dev_pull_async(dd->d_k_pc, tmp, size*sizeof(double), dd->stream); 
  }
  // Adding k_pc from all devices
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pm->dev_stream_wait(dd->stream);

    if(i > 0 && dd->d_k_pc) {
      
      tmp = &(buf_k_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) k_pc[j] += tmp[j];
    }
  }
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
#endif
}
/* ---------------------------------------------------------------------- */
void Device::pull_ints_ao2mo(py::array_t<double> _fxpp, py::array_t<double> _bufpa, int blksize, int naoaux, int nmo, int ncas)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
    py::buffer_info info_fxpp = _fxpp.request(); //3D array (nmo*nmo*naoaux)
    double * fxpp = static_cast<double*>(info_fxpp.ptr);
    //printf("size_fxpp %i\n", size_fxpp);
    
    int count = 0;
    int k = 0;

    // naive version to start; we can make this faster
    while(k < naoaux) {
      int size_vector = (naoaux-k > blksize) ? blksize : naoaux-k; // transfer whole blksize or last subset?
      
      //printf("k= %i  size_vector= %i\n",k,size_vector);
      for (int i=0; i<nmo; ++i)
	for (int j=0; j<nmo; ++j) {
	  int indx_in = count * nmo * nmo * blksize + i * nmo * size_vector + j * size_vector;
	  int indx_out = i * nmo * naoaux + j * naoaux + k;
	  
	  std::memcpy(&(fxpp[indx_out]), &(pin_fxpp[indx_in]), size_vector*sizeof(double));
	}
      
      k += blksize;
      count++;
    }
    
    py::buffer_info info_bufpa = _bufpa.request(); //3D array (naoaux*nmo*ncas)
    double * bufpa = static_cast<double*>(info_bufpa.ptr);
    //printf("size_bufpa %i\n", size_bufpa);
    std::memcpy(bufpa, pin_bufpa, size_bufpa*sizeof(double));

#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
#endif
}

/* ---------------------------------------------------------------------- */


double * Device::dd_fetch_eri(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
#if defined(_DEBUG_DEVICE) || defined(_DEBUG_ERI_CACHE)
  return dd_fetch_eri_debug(dd, eri1, naux, nao_pair, addr_dfobj, count);
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

double * Device::dd_fetch_eri_debug(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
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

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int blksize, int nset, int nao, int naux, int count)
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

  int nao_pair = nao * (nao+1) / 2;
  
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

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk, int nao, int nset, int with_k)
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

  int nao_pair = nao * (nao+1) / 2;
  
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
void Device::get_jk(int naux, int nao, int nset,
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
  
  printf("LIBGPU:: device= %i  naux= %i  nao= %i  nset= %i  nao_pair= %i  count= %i\n",device_id,naux,nao,nset,nao_pair,count);
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
    d_eri = dd_fetch_eri(dd, eri1, naux, nao_pair, addr_dfobj, count);

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
      
#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- get_jk::_getjk_rho :: nset= %i  naux= %i  RHO_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	     nset, naux, _RHO_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
    }
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)
   
    {
      dim3 grid_size(nset, (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, 1);
      dim3 block_size(1, _DOT_BLOCK_SIZE, 1);
      
      //      printf(" -- calling _getjk_vj()\n");
      int init = (count < num_devices) ? 1 : 0;
      _getjk_vj<<<grid_size, block_size, 0, dd->stream>>>(dd->d_vj, dd->d_rho, d_eri, nset, nao_pair, naux, init);

#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- get_jk::_getjk_vj :: nset= %i  nao_pair= %i _DOT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	     nset, nao_pair, _DOT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
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
#if 1
    dim3 grid_size(naux, _TILE(nao, _UNPACK_BLOCK_SIZE), 1);
    dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#else
    dim3 grid_size(naux, _TILE(nao*nao, _UNPACK_BLOCK_SIZE), 1);
    dim3 block_size(1, _UNPACK_BLOCK_SIZE, 1);
#endif
    
    _getjk_unpack_buf2<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf2, d_eri, dd->d_pumap_ptr, naux, nao, nao_pair);
    
#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- get_jk::_getjk_unpack_buf2 :: naux= %i  nao= %i _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
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
	printf("LIBGPU:: d_dms= %#012x  dms= %#012x  nao= %i  stream= %#012x\n",d_dms,dms,nao,dd->stream);
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
      dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, 1);
#else
      dim3 grid_size(naux*nao, 1, 1);
      dim3 block_size(1, _TRANSPOSE_BLOCK_SIZE, 1);
#endif
      
      _transpose<<<grid_size, block_size, 0, dd->stream>>>(dd->d_buf3, dd->d_buf1, naux*nao, nao);
      
#ifdef _DEBUG_DEVICE
      printf("LIBGPU ::  -- get_jk::_transpose :: naux= %i  nao= %i _TRANSPOSE_BLOCK_SIZE= %i  _TRANSPOSE_NUM_ROWS= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	     naux, nao, _TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_NUM_ROWS, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
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
__global__ void get_bufd( const double* bufpp, double* bufd, int naux, int nmo){
    
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < naux && j < nmo) {
        bufd[i * nmo + j] = bufpp[(i*nmo + j)*nmo + j];
    }
}

__global__ void get_bufpa (const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i<naux && j < nmo && k <ncas){
    int inputIndex = (i*nmo + j)*nmo + k+ncore;
    int outputIndex = (i*nmo + j)*ncas + k;
    bufpa[outputIndex]=bufpp[inputIndex];
    }
}

__global__ void get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao) {
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncas && j < nao) {
        small_mat[i * nao + j] = big_mat[j*nao + i+ncore];
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
void Device::df_ao2mo_pass1_v2 (int blksize, int nmo, int nao, int ncore, int ncas, int naux, 
				  py::array_t<double> _eri1,
				  int count, size_t addr_dfobj)
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif
  profile_start("AO2MO v2");
  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  //printf(" naux %i blksize %i\n", naux, blksize);
#ifdef _DEBUG_DEVICE
  printf("LIBGPU:: Inside Device::df_ao2mo_pass1_fdrv()\n");
  printf("LIBGPU:: dfobj= %#012x  count= %i  combined= %#012x %p update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count,addr_dfobj+count,update_dfobj);
#endif
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  const int nao_pair = nao*(nao+1)/2;
  double * eri = static_cast<double*>(info_eri1.ptr);
  
  int _size_eri = naux * nao_pair;
  int _size_eri_unpacked = naux * nao * nao; 
  
#ifdef _DEBUG_DEVICE
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif

  if(_size_eri_unpacked > dd->size_buf) {
    dd->size_buf = _size_eri_unpacked;
    
    if(dd->d_buf1) pm->dev_free_async(dd->d_buf1, dd->stream);
    if(dd->d_buf2) pm->dev_free_async(dd->d_buf2, dd->stream);
    
    dd->d_buf1 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double), dd->stream);// use for (eri@mo)
    dd->d_buf2 = (double *) pm->dev_malloc_async(dd->size_buf * sizeof(double), dd->stream);//use for eri_unpacked, then for bufpp_t
  }
  
  double * d_buf = dd->d_buf1; //for eri*mo_coeff (don't pull or push) 
  double * d_eri_unpacked = dd->d_buf2; //set memory for the entire eri array on GPU
  
  //unpack 2D eri of size naux * nao(nao+1)/2 to a full naux*nao*nao 3D matrix
  
  double * d_eri;
  
  if(use_eri_cache) {
    d_eri = dd_fetch_eri(dd, eri, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_eri > dd->size_eri1) {
      dd->size_eri1 = _size_eri;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1, dd->stream);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_eri * sizeof(double), dd->stream);
    }
    
    pm->dev_push(d_eri, eri, _size_eri * sizeof(double));
  }
  
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, nao);
  
  { dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    dim3 grid_size(_TILE(naux,block_size.x), _TILE(nao*nao, block_size.y), 1);
    _getjk_unpack_buf2<<<grid_size,block_size,0,dd->stream>>>(d_eri_unpacked, d_eri, my_d_tril_map_ptr, naux, nao, nao_pair); }
  
  //bufpp = mo.T @ eri @ mo
  //buf = np.einsum('ijk,kl->ijl',eri_unpacked,mo_coeff),i=naux,j=nao,l=nao
  double alpha = 1.0;
  double beta = 0.0;
  cublasDgemmStridedBatched(dd->handle, 
			    CUBLAS_OP_N, CUBLAS_OP_N,nao, nao, nao, 
			    &alpha,d_eri_unpacked, nao, nao*nao, dd->d_mo_coeff, nao, 0, 
			    &beta,d_buf, nao, nao*nao,naux);
  _CUDA_CHECK_ERRORS();
  
  //bufpp = np.einsum('jk,ikl->ijl',mo_coeff.T,buf),i=naux,j=nao,l=nao
  
  double * d_bufpp = dd->d_buf2;//set memory for the entire bufpp array, no pushing needed 
  cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, nao, 
			    &alpha, dd->d_mo_coeff, nao, 0, d_buf, nao, nao*nao, 
			    &beta, d_bufpp, nao, nao*nao,naux);
  _CUDA_CHECK_ERRORS();
#if 1
  int _size_bufpa = naux*nmo*ncas;
  if(_size_bufpa > dd->size_bufpa) {
    dd->size_bufpa = _size_bufpa;
    
    if(dd->d_bufpa) pm->dev_free_async(dd->d_bufpa, dd->stream);
    dd->d_bufpa = (double *) pm->dev_malloc_async(dd->size_bufpa * sizeof(double), dd->stream);
  }
  double * d_bufpa = dd->d_bufpa;
#else
  double * d_bufpa = (double *) pm->dev_malloc (naux*nmo*ncas*sizeof(double));
#endif

  { dim3 block_size(_UNPACK_BLOCK_SIZE,_UNPACK_BLOCK_SIZE,1);
    dim3 grid_size (_TILE(naux, block_size.x), _TILE(nmo, block_size.y), ncas);
    get_bufpa<<<grid_size, block_size, 0, dd->stream>>>(d_bufpp, d_bufpa, naux, nmo, ncore, ncas);}
#if 0
  py::buffer_info info_bufpa = _bufpa.request(); // 3D array (naux,nmo,ncas)
  double * bufpa = static_cast<double*>(info_bufpa.ptr);
#else
  double * bufpa = &(pin_bufpa[count*blksize*nmo*ncas]);
#endif
  pm->dev_pull_async(d_bufpa, bufpa, naux*nmo*ncas*sizeof(double), dd->stream);

  double * d_fxpp = dd->d_buf1;
  // fxpp[str(k)] =bufpp.transpose(1,2,0);
  { dim3 block_size (1, 1,1);
    dim3 grid_size (_TILE(naux, block_size.x),nmo,nmo) ;
    transpose_120<<<grid_size, block_size, 0, dd->stream>>>(d_bufpp, d_fxpp, naux, nmo, nmo);}

// calculate j_pc
  // k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])
if (count<num_devices)
    { cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                             1, 1, naux, 
                             &alpha,
                             d_fxpp, 1, naux,
                             d_fxpp, 1, naux, 
                             &beta,
                             dd->d_k_pc, 1, 1,
                             nmo*ncore); }
else
    { cublasDgemmStridedBatched(dd->handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                             1, 1, naux, 
                             &alpha,
                             d_fxpp, 1, naux,
                             d_fxpp, 1, naux, 
                             &alpha,
                             dd->d_k_pc, 1, 1,
                             nmo*ncore); }
#if 0
  py::buffer_info info_fxpp = _fxpp.request(); // 3D array (nmo,nmo, naux)
  double * fxpp = static_cast<double*>(info_fxpp.ptr);
#else
  double * fxpp = &(pin_fxpp[count*blksize*nmo*nmo]);
#endif


  pm->dev_pull_async(d_fxpp, fxpp, naux*nmo*nmo *sizeof(double), dd->stream);
  //bufd work
#if 1
  int _size_bufd = naux*nmo;
  if(_size_bufd > dd->size_bufd) {
    dd->size_bufd = _size_bufd;
    
    if(dd->d_bufd) pm->dev_free_async(dd->d_bufd, dd->stream);
    dd->d_bufd = (double *) pm->dev_malloc_async(dd->size_bufd * sizeof(double), dd->stream);
  }
  double * d_bufd = dd->d_bufd;
#else
    double * d_bufd = (double *) pm->dev_malloc_async(naux*nmo*sizeof(double),dd->stream);
#endif
  {dim3 block_size (_UNPACK_BLOCK_SIZE,_UNPACK_BLOCK_SIZE,1);
  dim3 grid_size (_TILE(naux, block_size.x),_TILE(nmo, block_size.y),1) ;
  get_bufd<<<grid_size, block_size, 0, dd->stream>>>(d_bufpp, d_bufd, naux, nmo);}
// calculate j_pc
  // self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
if (count<num_devices){
    cublasDgemm(dd->handle,CUBLAS_OP_N, CUBLAS_OP_T, 
               ncore, nmo, naux,
               &alpha, 
               d_bufd, nmo, 
               d_bufd, nmo, 
               &beta,
               dd->d_j_pc, ncore);}
else {
    cublasDgemm(dd->handle,CUBLAS_OP_N, CUBLAS_OP_T, 
               ncore, nmo, naux,
               &alpha, 
               d_bufd, nmo, 
               d_bufd, nmo, 
               &alpha,
               dd->d_j_pc, ncore);  }

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
  profile_stop(); 
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
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
  py::buffer_info info_h2eff_sub = _h2eff_sub.request();// 2d array (nmo * ncas) x (ncas*(ncas+1)/2)

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
  
  int _size_h2eff_unpacked = nmo*ncas*ncas*ncas;
  int _size_h2eff_packed = nmo*ncas*ncas_pair;

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
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE);
    dim3 grid_size(_TILE(ncas,block_size.x), _TILE(ncas,block_size.y));
    
    extract_submatrix<<<grid_size, block_size, 0, dd->stream>>>(dd->d_umat, dd->d_ucas, ncas, ncore, nmo);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- update_h2eff_sub::extract_submatrix :: ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
  //h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
  //h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)

  if(_size_h2eff_packed > dd->size_h2eff) {
    dd->size_h2eff = _size_h2eff_packed;
    if(dd->d_h2eff) pm->dev_free_async(dd->d_h2eff, dd->stream);
    dd->d_h2eff = (double *) pm->dev_malloc_async(dd->size_h2eff * sizeof(double), dd->stream);
  }
  
  double * d_h2eff_sub = dd->d_h2eff;
  
  pm->dev_push_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed * sizeof(double), dd->stream);

  profile_next("map creation and pushed");
  
  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_UNPACK);

  profile_next("unpacking");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- created and pushed unpacking map\n");
#endif
  
  {
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    dim3 grid_size(_TILE(nmo*ncas,_UNPACK_BLOCK_SIZE), _TILE(ncas*ncas,_UNPACK_BLOCK_SIZE), 1);
    
    _unpack_h2eff_2d<<<grid_size, block_size, 0, dd->stream>>>(d_h2eff_sub, d_h2eff_unpacked, d_my_unpack_map_ptr, nmo, ncas, ncas_pair);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- update_h2eff_sub::_unpack_h2eff_2d :: nmo*ncas= %i  ncas*ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   nmo*ncas, ncas*ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
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
  
  //h2eff_tranposed=(piJK->JKpi)
  
  double * d_h2eff_transposed = dd->d_buf2;
  
  {
    dim3 block_size(1,1,_DEFAULT_BLOCK_SIZE);
    dim3 grid_size(_TILE(nmo,block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));

    transpose_2310<<<grid_size, block_size, 0, dd->stream>>>(d_h2eff_step2, d_h2eff_transposed, nmo, ncas);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- update_h2eff_sub::transpose_2310 :: nmo= %i  ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   nmo, ncas, _DEFAULT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
  profile_next("last 2 dgemm");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished transposing\n");
#endif
  
  double * d_h2eff_step3 = dd->d_buf1;

  //h2eff_sub=np.einsum('iI,JKip->JKIp',ucas,h2eff_sub) h2eff=ncas,ncas,ncas,nmo; ucas=ncas,ncas

  cublasDgemmStridedBatched(dd->handle,CUBLAS_OP_N,CUBLAS_OP_T,nmo,ncas,ncas,
			    &alpha, 
			    d_h2eff_transposed, nmo, ncas*nmo, 
			    dd->d_ucas, ncas, 0, 
			    &beta, d_h2eff_step3, nmo, ncas*nmo, ncas*ncas);
  _CUDA_CHECK_ERRORS();

  //h2eff_step4=([JK]Ip,pP->[JK]IP)

  double * d_h2eff_step4 = dd->d_buf2;
  
  cublasDgemmStridedBatched(dd->handle,CUBLAS_OP_N,CUBLAS_OP_N,nmo,ncas,nmo,
			    &alpha, 
			    dd->d_umat, nmo, 0, 
			    d_h2eff_step3, nmo, ncas*nmo, 
			    &beta, d_h2eff_step4, nmo, ncas*nmo, ncas*ncas);
  
  profile_next("2nd transpose");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished last 2 cublasDgemmStridedBatched Functions \n");
#endif

  double * d_h2eff_transpose2 = dd->d_buf1;
  
  //h2eff_tranposed=(JKIP->PIJK) 3201

  {
    dim3 block_size(1,1,_DEFAULT_BLOCK_SIZE);
    dim3 grid_size(_TILE(ncas,block_size.x),_TILE(ncas,block_size.y),_TILE(ncas,block_size.z));
    
    transpose_3210<<<grid_size, block_size, 0, dd->stream>>>(d_h2eff_step4, d_h2eff_transpose2, nmo, ncas);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- update_h2eff_sub::transpose_3210 :: ncas= %i  _DEFAULT_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   ncas, _DEFAULT_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
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
    dim3 block_size(1, 1, _UNPACK_BLOCK_SIZE);
    dim3 grid_size(nmo, ncas, _TILE(ncas_pair, _DEFAULT_BLOCK_SIZE));
    
    _pack_h2eff_2d<<<grid_size, block_size, 0, dd->stream>>>(d_h2eff_transpose2, d_h2eff_sub, d_my_pack_map_ptr, nmo, ncas, ncas_pair);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- update_h2eff_sub::_pack_h2eff_2d :: nmo= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   nmo, ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Freed map\n");
#endif
  
  pm->dev_pull_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed*sizeof(double), dd->stream);

  pm->dev_stream_wait(dd->stream);
  
  profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Leaving update function\n");
  cudaMemGetInfo(&freeMem, &totalMem);
  
  printf("Ending h2eff_sub_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
  
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[5] += t1 - t0;
#endif
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

  {
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    dim3 grid_size(_TILE(naux,_UNPACK_BLOCK_SIZE), _TILE(nao,_UNPACK_BLOCK_SIZE));
    
    _getjk_unpack_buf2<<<grid_size,block_size, 0, dd->stream>>>(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- h2eff_df_contract1::_getjk_unpack_buf2 :: naux= %i  nao= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
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
  {
    dim3 block_size(_UNPACK_BLOCK_SIZE, 1, _UNPACK_BLOCK_SIZE);
    dim3 grid_size(_TILE(naux,block_size.x), _TILE(ncas,block_size.y), _TILE(nao,block_size.z));
    
    transpose_210<<<grid_size,block_size, 0, dd->stream>>>(d_bPmu, d_bmuP,naux,nao,ncas);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- h2eff_df_contract1::transpose_210 :: naux= %i  ncas= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   naux, ncas, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
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
  t_array[6] += t1 - t0;
#endif
}
/* ---------------------------------------------------------------------- */
__global__ void pack_d_vuwM(const double * in, double * out, int nmo, int ncas, int ncas_pair, int * map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= nmo*ncas) return;
    if(j >= ncas*ncas) return;
    //out[k*ncas_pair*nao+l*ncas_pair+ij]=h_vuwM[i*ncas*ncas*nao+j*ncas*nao+k*nao+l];}}}}
    out[i*ncas_pair + map[j]]=in[j*ncas*nmo + i];
}

/* ---------------------------------------------------------------------- */
void Device::get_h2eff_df(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri, int count, size_t addr_dfobj) 
{
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting h2eff_df_contract1 function");
  printf("LIBGPU:: dfobj= %#012x count= %i combined= %#012x %p update_dfobj= %i\n",addr_dfobj,count,addr_dfobj+count,addr_dfobj+count,update_dfobj);
#endif 
  
  profile_start("h2eff df setup");
  
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas
  
  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);
  
  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri = nmo*ncas*ncas_pair;
  const int _size_cderi = naux*nao_pair;
  const int _size_cderi_unpacked = naux*nao*nao;
  const int _size_mo_cas = nao*ncas;
  double * eri = static_cast<double*>(info_eri.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  double * d_mo_cas = (double*) pm->dev_malloc(_size_mo_cas*sizeof(double));
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  double * cderi = static_cast<double*>(info_cderi.ptr);

  // d_mo_cas
  {
    dim3 block_size(1,1,1);
    dim3 grid_size(_TILE(ncas, block_size.x), _TILE(nao, block_size.y));
    
    get_mo_cas<<<grid_size, block_size, 0, dd->stream>>>(d_mo_coeff, d_mo_cas, ncas, ncore, nao);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- get_h2eff_df::_get_mo_cas :: ncas= %i  nao= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   ncas, nao, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }

  double * d_cderi;
  if(use_eri_cache) {
    d_cderi = dd_fetch_eri(dd, cderi, naux, nao_pair, addr_dfobj, count);
  } else {
    if(_size_cderi > dd->size_eri1) {
      dd->size_eri1 = _size_cderi;
      if(dd->d_eri1) pm->dev_free_async(dd->d_eri1, dd->stream);
      dd->d_eri1 = (double *) pm->dev_malloc_async(_size_cderi * sizeof(double), dd->stream);
    }
    
    pm->dev_push(d_cderi, cderi, _size_cderi * sizeof(double));
  }

  double * d_cderi_unpacked=(double*) pm->dev_malloc( _size_cderi_unpacked * sizeof(double));

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao);

  {
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    dim3 grid_size(_TILE(naux,_UNPACK_BLOCK_SIZE), _TILE(nao,_UNPACK_BLOCK_SIZE));
    
    _getjk_unpack_buf2<<<grid_size,block_size, 0, dd->stream>>>(d_cderi_unpacked,d_cderi,d_my_unpack_map_ptr,naux, nao, nao_pair);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- get_h2eff_df::_getjk_unpack_buf2 :: naux= %i  nao= %i  _UNPACK_BLOCK_SIZE= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   naux, nao, _UNPACK_BLOCK_SIZE, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
//bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  const double alpha=1.0;
  const double beta=0.0;
  const int _size_bPmu = naux*ncas*nao;
  double * d_bPmu = (double*) pm->dev_malloc (_size_bPmu *sizeof(double));
  cublasDgemmStridedBatched (dd->handle,CUBLAS_OP_N, CUBLAS_OP_N, 
                              nao, ncas, nao,
                              &alpha, d_cderi_unpacked, nao, nao*nao, d_mo_cas, nao, 0,
                              &beta, d_bPmu, nao, ncas*nao, naux);
  _CUDA_CHECK_ERRORS();

  pm->dev_free(d_cderi_unpacked);

//bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)
  const int _size_bPvu = naux*ncas*ncas;
  double * d_bPvu = (double*) pm->dev_malloc (_size_bPvu *sizeof(double));
  cublasDgemmStridedBatched (dd->handle, CUBLAS_OP_C, CUBLAS_OP_N, 
                             ncas, ncas, nao, 
                             &alpha, d_mo_cas, nao, 0,
                             d_bPmu, nao, ncas*nao, 
                             &beta, d_bPvu, ncas, ncas*ncas, naux); 

//eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)
  //transpose bPmu 
  double * d_bumP = (double*) pm->dev_malloc (_size_bPmu *sizeof(double));
  {
    dim3 block_size(1,1,1);
    dim3 grid_size(_TILE(naux, block_size.x),_TILE(nao, block_size.y),_TILE(ncas, block_size.z));
    
    transpose_120<<<grid_size, block_size, 0, dd->stream>>>(d_bPmu, d_bumP, naux, ncas, nao);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- get_h2eff_df::transpose_120 :: naux= %i  nao= %i  ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   naux, nao, ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
  pm->dev_free(d_bPmu);

  double * d_buvP = (double*) pm->dev_malloc (_size_bPvu *sizeof(double));
  //transpose bPvu
  {
    dim3 block_size(1,1,1);
    dim3 grid_size(_TILE(naux, block_size.x),_TILE(ncas, block_size.y),_TILE(ncas, block_size.z));
    
    transpose_210<<<grid_size, block_size, 0, dd->stream>>>(d_bPvu, d_buvP, naux, ncas, ncas);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- get_h2eff_df::transpose_210 :: naux= %i  ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   naux, ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }

  pm->dev_free(d_bPvu);

  //h_vuwm[i*ncas*nao+j]+=h_bvuP[i*naux + k]*h_bumP[j*naux+k];
  //dgemm (probably just simple, not strided/batched, contracted dimension = P)
  const int _size_mwvu = nao*ncas*ncas*ncas;
  double * d_vuwm = (double*) pm ->dev_malloc( _size_mwvu*sizeof(double));
cublasDgemm(dd->handle, CUBLAS_OP_T, CUBLAS_OP_N,
              ncas*nao, ncas*ncas, naux,
              &alpha, 
              d_bumP, naux,
              d_buvP, naux,
              &beta, 
              d_vuwm, ncas*nao);

  pm->dev_free(d_bumP);
  pm->dev_free(d_buvP);

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
  pm->dev_free(d_vuwm);

  double * d_eri = (double*) pm->dev_malloc (_size_eri*sizeof(double));
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, ncas);

  {
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    dim3 grid_size(_TILE(nmo*ncas,block_size.x), _TILE(ncas*ncas,block_size.y));
    
    pack_d_vuwM<<<grid_size,block_size, 0, dd->stream>>>(d_vuwM,d_eri,nmo, ncas, ncas_pair, my_d_tril_map_ptr);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU ::  -- get_h2eff_df::pack_d_vumM :: nmo*ncas= %i  ncas*ncas= %i  grid_size= %i %i %i  block_size= %i %i %i\n",
	   nmo*ncas, ncas*ncas, grid_size.x,grid_size.y,grid_size.z,block_size.x,block_size.y,block_size.z);
    _CUDA_CHECK_ERRORS();
#endif
  }
  
  pm->dev_free(d_vuwM);

  pm->dev_pull(d_eri, eri, _size_eri*sizeof(double));

  pm->dev_free(d_eri);

profile_stop();
#ifdef _SIMPLE_TIMER
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;//TODO: add the array size
#endif
}
#endif
