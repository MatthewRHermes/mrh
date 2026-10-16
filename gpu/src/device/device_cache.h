/* -*- c++ -*- */

#ifndef DEVICE_CACHE_H
#define DEVICE_CACHE_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Shared ERI-block and pack/unpack-map cache.
//
// Owns the ERI cache bookkeeping (eri_list/count/update/size/num_blocks/extra/
// device + the d_eri_cache/d_eri_host device pointers) plus the update_dfobj
// flag. The pack/unpack maps themselves are stored
// per-device in my_device_data::fci (borrowed via ctx.device_data); only the
// fetch/allocate logic lives here. Everything else is borrowed through the
// DeviceContext provided by the Device facade.
//
// It used to live on Device and was reached from subdomains via
// ctx.owner->dd_fetch_eri / dd_fetch_pumap; it is now an owned subdomain
// reachable via ctx.cache->dd_fetch_eri / dd_fetch_pumap.
// The former ctx.owner back-reference no longer exists.
class DeviceCache {

public:

  DeviceCache(DeviceContext & ctx);
  ~DeviceCache();

  void set_update_dfobj_(int _val);
  void get_dfobj_status(size_t addr_dfobj, py::array_t<int> _arg);

  int * dd_fetch_pumap(my_device_data *, int, int);
  double * dd_fetch_eri(my_device_data *, double *, int, int, size_t, int);
  double * dd_fetch_eri_debug(my_device_data *, double *, int, int, size_t, int);

  int update_dfobj; // reset to zero in DeviceJk::pull_get_jk

private:

  DeviceContext & ctx;

  std::vector<size_t> eri_list; // addr of dfobj+eri1 for key-value pair

  std::vector<int> eri_count; // # times particular cache used
  std::vector<int> eri_update; // # times particular cache updated
  std::vector<int> eri_size; // # size of particular cache

  std::vector<int> eri_num_blocks; // # of eri blocks for each dfobj (i.e. trip-count from `for eri1 in dfobj.loop(blksize)`)
  std::vector<int> eri_extra; // per-block data: {naux, nao_pair}
  std::vector<int> eri_device; // device id holding cache

  std::vector<double *> d_eri_cache; // pointers for device caches
  std::vector<double *> d_eri_host; // values on host for checking if update
};

#endif
