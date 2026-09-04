/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#include <unistd.h>
#include <string.h>
#include <sched.h>
#define _MIN(A,B) (A<B)?A:B
#define _MAX(A,B) (A>B)?A:B
#define _SIZE_FCI_BATCHES 6

/* ---------------------------------------------------------------------- */

DeviceCache::DeviceCache(DeviceContext & _ctx)
  : ctx(_ctx)
{
  update_dfobj = 0;
}

DeviceCache::~DeviceCache()
{
  if(ctx.verbose_level) {
    printf("\nLIBGPU :: eri cache statistics :: count= %zu\n",eri_list.size());
    for(int i=0; i<eri_list.size(); ++i)
      printf("LIBGPU :: %i : eri= %p  Mbytes= %f  count= %i  update= %i device= %i\n", i, (void*) eri_list[i],
	     eri_size[i]*sizeof(double)/1024./1024., eri_count[i], eri_update[i], eri_device[i]);
  }

  eri_count.clear();
  eri_size.clear();
#ifdef _DEBUG_ERI_CACHE
  for(int i=0; i<d_eri_host.size(); ++i) ctx.pm->dev_free_host( d_eri_host[i] );
#endif
  for(int i=0; i<d_eri_cache.size(); ++i) {
    int id = eri_device[i];
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_free(d_eri_cache[i], "eri_cache");
  }
  eri_list.clear();
}

/* ---------------------------------------------------------------------- */

void DeviceCache::set_update_dfobj_(int _val)
{
  update_dfobj = _val; // this is reset to zero in Device::pull_get_jk
}

/* ---------------------------------------------------------------------- */

// return stored values for Python side to make decisions
// update_dfobj == true :: nothing useful to return if need to update eri blocks on device
// count_ == -1 :: return # of blocks cached for dfobj
// count_ >= 0 :: return extra data for cached block

void DeviceCache::get_dfobj_status(size_t addr_dfobj, py::array_t<int> _arg)
{
  py::buffer_info info_arg = _arg.request();
  int * arg = static_cast<int*>(info_arg.ptr);

  int naux_ = arg[0];
  int nao_pair_ = arg[1];
  int count_ = arg[2];
  int update_dfobj_ = arg[3];

  // printf("Inside get_dfobj_status(): addr_dfobj= %#012x  naux_= %i  nao_pair_= %i  count_= %i  update_dfobj_= %i\n",
  // 	 addr_dfobj, naux_, nao_pair_, count_, update_dfobj_);

  update_dfobj_ = update_dfobj;

  // nothing useful to return if need to update eri blocks on device

  if(update_dfobj) {
    // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  update_dfobj_= %i\n", addr_dfobj, update_dfobj_);

    arg[3] = update_dfobj_;
    return;
  }

  // return # of blocks cached for dfobj

  if(count_ == -1) {
    int id = eri_list.size();
    for(int i=0; i<eri_list.size(); ++i)
      if(eri_list[i] == addr_dfobj) {
	id = i;
	break;
      }

    if(id < eri_list.size()) count_ = eri_num_blocks[id];

    // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  count_= %i  update_dfobj_= %i\n", addr_dfobj, count_, update_dfobj_);

    arg[2] = count_;
    arg[3] = update_dfobj_;
    return;
  }

  // return extra data for cached block

  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count_) {
      id = i;
      break;
    }

  // printf("eri_list.size()= %i  id= %i\n",eri_list.size(), id);

  naux_ = -1;
  nao_pair_ = -1;

  if(id < eri_list.size()) {

    naux_     = eri_extra[id * _ERI_CACHE_EXTRA    ];
    nao_pair_ = eri_extra[id * _ERI_CACHE_EXTRA + 1];

  }

  arg[0] = naux_;
  arg[1] = nao_pair_;
  arg[2] = count_;
  arg[3] = update_dfobj_;

  // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  id= %i  naux_= %i  nao_pair_= %i  count_= %i  update_dfobj_= %i\n",
  // 	 addr_dfobj, id, naux_, nao_pair_, count_, update_dfobj_);

  // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  id= %i  arg= %i %i %i %i\n",
  // 	 addr_dfobj, id, arg[0], arg[1], arg[2], arg[3]);
}

/* ---------------------------------------------------------------------- */

int * DeviceCache::dd_fetch_pumap(my_device_data * dd, int size_pumap_, int type_pumap)
{
  // search if pack/unpack map already created

  int indx = -1;
  for(int i=0; i<dd->fci.size_pumap.size(); ++i)
    if(dd->fci.type_pumap[i] == type_pumap && dd->fci.size_pumap[i] == size_pumap_) indx = i;

  // add unpack/pack map if not found

  if(indx < 0) {
    dd->fci.type_pumap.push_back(type_pumap);
    dd->fci.size_pumap.push_back(size_pumap_);
    dd->fci.pumap.push_back(nullptr);
    dd->fci.d_pumap.push_back(nullptr);

    indx = dd->fci.type_pumap.size() - 1;

    int size_pumap = -1;

    if(type_pumap == _PUMAP_2D_UNPACK) {
      int nao = size_pumap_;
      size_pumap = nao * nao;

      dd->fci.pumap[indx] = (int *) ctx.pm->dev_malloc_host(size_pumap * sizeof(int));

      std::string name = "pumap-" + std::to_string(indx);
      dd->fci.d_pumap[indx] = (int *) ctx.pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

      int _i, _j, _ij;
      int * tm = dd->fci.pumap[indx];
      for(_ij = 0, _i = 0; _i < nao; _i++)
	for(_j = 0; _j<=_i; _j++, _ij++) {
	  tm[_i*nao + _j] = _ij;
	  tm[_i + nao*_j] = _ij;
	}

    } else if(type_pumap == _PUMAP_H2EFF_UNPACK) {

#if 1
      int ncas = size_pumap_;
      size_pumap = ncas * ncas;

      dd->fci.pumap[indx] = (int *) ctx.pm->dev_malloc_host(size_pumap * sizeof(int));

      std::string name = "pumap-" + std::to_string(indx);
      dd->fci.d_pumap[indx] = (int *) ctx.pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

      int * tm = dd->fci.pumap[indx];
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

      dd->fci.pumap[indx] = (int *) ctx.pm->dev_malloc_host(size_pumap * sizeof(int));

      std::string name = "pumap-" + std::to_string(indx);
      dd->fci.d_pumap[indx] = (int *) ctx.pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

      int * tm = dd->fci.pumap[indx];
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

      dd->fci.pumap[indx] = (int *) ctx.pm->dev_malloc_host(size_pumap * sizeof(int));

      std::string name = "pumap-" + std::to_string(indx);
      dd->fci.d_pumap[indx] = (int *) ctx.pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

      int * tm = dd->fci.pumap[indx];
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

      dd->fci.pumap[indx] = (int *) ctx.pm->dev_malloc_host(size_pumap * sizeof(int));

      std::string name = "pumap-" + std::to_string(indx);
      dd->fci.d_pumap[indx] = (int *) ctx.pm->dev_malloc_async(size_pumap * sizeof(int), name, FLERR);

      int * tm = dd->fci.pumap[indx];
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
    ctx.pm->dev_push_async(dd->fci.d_pumap[indx], dd->fci.pumap[indx], size_pumap*sizeof(int));
  } // if(map_not_found)

  // set pointers to current map

  dd->fci.pumap_ptr = dd->fci.pumap[indx];
  dd->fci.d_pumap_ptr = dd->fci.d_pumap[indx];

  return dd->fci.d_pumap_ptr;
}

/* ---------------------------------------------------------------------- */

double * DeviceCache::dd_fetch_eri(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
#if defined(_DEBUG_DEVICE) || defined(_DEBUG_ERI_CACHE)
  if(eri1 != nullptr) return dd_fetch_eri_debug(dd, eri1, naux, nao_pair, addr_dfobj, count);
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

    bool full_molecule = id-count == 0;

    if(!full_molecule && update_dfobj) {
      eri_update[id]++;
      int err = ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
      if(err) {
	printf("LIBGPU:: dev_push_async(d_eri) updating eri block\n");
	exit(1);
      }
    }

    if(naux != eri_extra[id*2] || nao_pair != eri_extra[id*2+1]) {
      printf("LIBGPU :: dd_fetch_eri() has inconsistent naux= {%i, %i} and nao_pair= {%i, %i} for block id= %i\n",naux, eri_extra[id*2], nao_pair, eri_extra[id*2+1], id);
      exit(1);
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

    d_eri = (double *) ctx.pm->dev_malloc_async(naux * nao_pair * sizeof(double), "eri_cache", FLERR);
    d_eri_cache.push_back(d_eri);

    int err = ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
      exit(1);
    }

#ifdef _DEBUG_DEVICE
    printf("LIBGPU:: dd_fetch_eri :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",(void*)(addr_dfobj+count), count, naux, nao_pair);
#endif
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */

double * DeviceCache::dd_fetch_eri_debug(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting eri_cache lookup for ERI %p\n",(void*)(addr_dfobj+count));
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
      ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
      eri_update[id]++;

      // update_dfobj fails to correctly update device ; this is an error
      if(!update_dfobj) {
	printf("LIBGPU :: Warning: ERI %p updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",(void*)(addr_dfobj+count),diff_eri,update_dfobj);
	//count = -1;
	//return;
	exit(1);
      }
    } else {

      // update_dfobj falsely updates device ; this is loss of performance
      if(update_dfobj) {
	printf("LIBGPU :: Warning: ERI %p not updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",(void*)(addr_dfobj+count)//,diff_eri,update_dfobj);
	//count = -1;
	//return;
	//exit(1);
      }
    }
#else
    if(update_dfobj) {
#ifdef _DEBUG_DEVICE
      printf("LIBGPU :: -- updating eri block: id= %i\n",id);
#endif
      eri_update[id]++;
      int err = ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
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

    int id_ = d_eri_cache.size();
#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- allocating new eri block: %i\n",id);
#endif

    d_eri = (double *) ctx.pm->dev_malloc_async(naux * nao_pair * sizeof(double), "eri_cache", FLERR);
    d_eri_cache.push_back(d_eri);

#ifdef _DEBUG_DEVICE
    printf("LIBGPU :: -- initializing eri block\n");
#endif
    int err = ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
      exit(1);
    }

#ifdef _DEBUG_ERI_CACHE
    d_eri_host.push_back( (double *) ctx.pm->dev_malloc_host(naux*nao_pair * sizeof(double)) );
    double * d_eri_host_ = d_eri_host[id_];
    for(int i=0; i<naux*nao_pair; ++i) d_eri_host_[i] = eri1[i];
#endif

#ifdef _DEBUG_DEVICE
    printf("LIBGPU:: dd_fetch_eri_debug :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",(void*)(addr_dfobj+count), count, naux, nao_pair);
#endif
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */
