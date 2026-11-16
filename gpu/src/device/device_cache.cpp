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
  full_dfobj = 0;
}

DeviceCache::~DeviceCache()
{
  if(ctx.verbose_level) {
    printf("\nLIBGPU :: eri cache statistics :: count= %zu  evicted= %i\n",eri_addr.size(), num_evicted);
    for(int i=0; i<eri_addr.size(); ++i)
      printf("LIBGPU :: %i : dfobj= %p  block= %i  Mbytes= %f  count= %i  update= %i device= %i\n", i, (void*) eri_addr[i], eri_block[i],
	     eri_size[i]*sizeof(double)/1024./1024., eri_count[i], eri_update[i], eri_device[i]);
  }

#ifdef _DEBUG_ERI_CACHE
  for(int i=0; i<d_eri_host.size(); ++i) ctx.pm->dev_free_host( d_eri_host[i] );
#endif
  for(int i=0; i<d_eri_cache.size(); ++i) {
    int id = eri_device[i];
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_free(d_eri_cache[i], "eri_cache");
  }
  eri_addr.clear();
}

/* ---------------------------------------------------------------------- */

void DeviceCache::set_update_dfobj_(int _val)
{
  update_dfobj = _val; // this is reset to zero in DeviceJk::pull_get_jk
}

/* ---------------------------------------------------------------------- */

// index of cached block {addr_dfobj, count}, or -1 if not cached
//  - keyed on the pair: addr_dfobj+count can collide across dfobjs with many blocks

int DeviceCache::find_eri(size_t addr_dfobj, int count)
{
  for(int i=0; i<eri_addr.size(); ++i)
    if(eri_addr[i] == addr_dfobj && eri_block[i] == count) return i;
  return -1;
}

/* ---------------------------------------------------------------------- */

// drop (and free) all cached blocks of dfobj; returns # of blocks dropped
//  - called when the DF object discards its integrals (DF.reset, e.g. geometry change)
//    and when it is garbage collected (id(dfobj) may be reused by a new object)

int DeviceCache::evict_dfobj(size_t addr_dfobj)
{
  int nevict = 0;
  const int device_id = ctx.pm->dev_get_device();

  for(int i=0; i<eri_addr.size(); ) {
    if(eri_addr[i] != addr_dfobj) { ++i; continue; }

    ctx.pm->dev_set_device(eri_device[i]);
    ctx.pm->dev_free(d_eri_cache[i], "eri_cache");
#ifdef _DEBUG_ERI_CACHE
    ctx.pm->dev_free_host(d_eri_host[i]);
    d_eri_host.erase(d_eri_host.begin() + i);
#endif

    eri_addr.erase(eri_addr.begin() + i);
    eri_block.erase(eri_block.begin() + i);
    eri_count.erase(eri_count.begin() + i);
    eri_update.erase(eri_update.begin() + i);
    eri_size.erase(eri_size.begin() + i);
    eri_device.erase(eri_device.begin() + i);
    eri_extra.erase(eri_extra.begin() + i*_ERI_CACHE_EXTRA, eri_extra.begin() + (i+1)*_ERI_CACHE_EXTRA);
    d_eri_cache.erase(d_eri_cache.begin() + i);

    nevict++;
  }

  if(nevict) ctx.pm->dev_set_device(device_id);
  if(addr_dfobj == full_dfobj) full_dfobj = 0;
  num_evicted += nevict;

  return nevict;
}

/* ---------------------------------------------------------------------- */

// return stored values for Python side to make decisions
// update_dfobj == true :: nothing useful to return if need to update eri blocks on device
// count_ == -1 :: return # of blocks cached for dfobj (-1 if none)
// count_ >= 0 :: return extra data for cached block (-1 if not cached)

void DeviceCache::get_dfobj_status(size_t addr_dfobj, py::array_t<int> _arg)
{
  py::buffer_info info_arg = _arg.request();
  int * arg = static_cast<int*>(info_arg.ptr);

  int count_ = arg[2];

  arg[3] = update_dfobj;

  // nothing useful to return if need to update eri blocks on device

  if(update_dfobj) return;

  // return # of blocks cached for dfobj

  if(count_ == -1) {
    int nblk = 0;
    for(int i=0; i<eri_addr.size(); ++i) if(eri_addr[i] == addr_dfobj) nblk++;
    arg[2] = (nblk > 0) ? nblk : -1;
    return;
  }

  // return extra data for cached block

  int id = find_eri(addr_dfobj, count_);

  arg[0] = (id < 0) ? -1 : eri_extra[id * _ERI_CACHE_EXTRA    ];
  arg[1] = (id < 0) ? -1 : eri_extra[id * _ERI_CACHE_EXTRA + 1];
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

// register a new block; full_dfobj is the first dfobj cached into an empty cache (the full molecule in LAS)

double * DeviceCache::insert_eri(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
  if(eri1 == nullptr) {
    printf("LIBGPU :: dd_fetch_eri() no cached block for dfobj= %p  block= %i and no host data to fill it;"
	   " the DF object was reset or never ran get_jk on this device\n", (void*) addr_dfobj, count);
    exit(1);
  }

  if(eri_addr.empty()) full_dfobj = addr_dfobj;

  eri_addr.push_back(addr_dfobj);
  eri_block.push_back(count);
  eri_count.push_back(1);
  eri_update.push_back(0);
  eri_size.push_back(naux * nao_pair);
  eri_device.push_back(dd->device_id);

  eri_extra.push_back(naux);
  eri_extra.push_back(nao_pair);

  double * d_eri = (double *) ctx.pm->dev_malloc_async(naux * nao_pair * sizeof(double), "eri_cache", FLERR);
  d_eri_cache.push_back(d_eri);

  int err = ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
  if(err) {
    printf("LIBGPU:: dev_push_async(d_eri) initializing new eri block\n");
    exit(1);
  }

#ifdef _DEBUG_ERI_CACHE
  d_eri_host.push_back( (double *) ctx.pm->dev_malloc_host(naux*nao_pair * sizeof(double)) );
  double * d_eri_host_ = d_eri_host.back();
  for(int i=0; i<naux*nao_pair; ++i) d_eri_host_[i] = eri1[i];
#endif

#ifdef _DEBUG_DEVICE
  printf("LIBGPU:: dd_fetch_eri :: addr= %p  count= %i  naux= %i  nao_pair= %i\n",(void*)addr_dfobj, count, naux, nao_pair);
#endif

  return d_eri;
}

/* ---------------------------------------------------------------------- */

double * DeviceCache::dd_fetch_eri(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
#if defined(_DEBUG_DEVICE) || defined(_DEBUG_ERI_CACHE)
  if(eri1 != nullptr) return dd_fetch_eri_debug(dd, eri1, naux, nao_pair, addr_dfobj, count);
#endif

  int id = find_eri(addr_dfobj, count);

  if(id < 0) return insert_eri(dd, eri1, naux, nao_pair, addr_dfobj, count);

  eri_count[id]++;
  double * d_eri = d_eri_cache[id];

  // update_dfobj == 1 :: refresh blocks of sub-fragment dfobjs (rebuilt in place by LAS); the full-molecule dfobj is fixed
  // integrals of a dfobj that change otherwise (e.g. geometry change) are evicted from the cache via DF.reset

  const bool full_molecule = (addr_dfobj == full_dfobj);
  const bool refresh = update_dfobj && !full_molecule;

  if(naux != eri_extra[id*2] || nao_pair != eri_extra[id*2+1]) {
    if(!refresh) {
      printf("LIBGPU :: dd_fetch_eri() has inconsistent naux= {%i, %i} and nao_pair= {%i, %i} for dfobj= %p block= %i\n",
	     naux, eri_extra[id*2], nao_pair, eri_extra[id*2+1], (void*) addr_dfobj, count);
      exit(1);
    }

    // refreshed block changed shape: reallocate
    ctx.pm->dev_free(d_eri, "eri_cache");
    d_eri = (double *) ctx.pm->dev_malloc_async(naux * nao_pair * sizeof(double), "eri_cache", FLERR);
    d_eri_cache[id] = d_eri;
    eri_size[id] = naux * nao_pair;
    eri_extra[id*2] = naux;
    eri_extra[id*2+1] = nao_pair;
  }

  if(refresh) {
    eri_update[id]++;
    int err = ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    if(err) {
      printf("LIBGPU:: dev_push_async(d_eri) updating eri block\n");
      exit(1);
    }
  }

  return d_eri;
}

/* ---------------------------------------------------------------------- */

double * DeviceCache::dd_fetch_eri_debug(my_device_data * dd, double * eri1, int naux, int nao_pair, size_t addr_dfobj, int count)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Starting eri_cache lookup for dfobj= %p block= %i\n",(void*)addr_dfobj, count);
#endif

  int id = find_eri(addr_dfobj, count);

  if(id < 0) return insert_eri(dd, eri1, naux, nao_pair, addr_dfobj, count);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- eri block found: id= %i\n",id);
#endif

  eri_count[id]++;
  double * d_eri = d_eri_cache[id];

  if(naux != eri_extra[id*2] || nao_pair != eri_extra[id*2+1]) {
    printf("LIBGPU :: dd_fetch_eri_debug() has inconsistent naux= {%i, %i} and nao_pair= {%i, %i} for dfobj= %p block= %i\n",
	   naux, eri_extra[id*2], nao_pair, eri_extra[id*2+1], (void*) addr_dfobj, count);
    exit(1);
  }

#ifdef _DEBUG_ERI_CACHE
  double * eri_host = d_eri_host[id];
  double diff_eri = 0.0;
  for(int i=0; i<naux*nao_pair; ++i) diff_eri += (eri_host[i] - eri1[i]) * (eri_host[i] - eri1[i]);

  if(diff_eri > 1e-10) {
    for(int i=0; i<naux*nao_pair; ++i) eri_host[i] = eri1[i];
    ctx.pm->dev_push_async(d_eri, eri1, naux * nao_pair * sizeof(double));
    eri_update[id]++;

    // update_dfobj fails to correctly update device ; this is an error
    if(!update_dfobj) {
      printf("LIBGPU :: Warning: ERI dfobj= %p block= %i updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",(void*)addr_dfobj,count,diff_eri,update_dfobj);
      exit(1);
    }
  } else {

    // update_dfobj falsely updates device ; this is loss of performance
    if(update_dfobj) {
      printf("LIBGPU :: Warning: ERI dfobj= %p block= %i not updated on device w/ diff_eri= %.10e, but update_dfobj= %i\n",(void*)addr_dfobj,count,diff_eri,update_dfobj);
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

  return d_eri;
}
