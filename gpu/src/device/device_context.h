/* -*- c++ -*- */

#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

#include <string>
#include <vector>
#include <unordered_map>

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"

class DeviceComm; // forward decl for DeviceContext::comm (owned multi-GPU comm subdomain)
class DeviceCache; // forward decl for DeviceContext::cache (owned ERI cache subdomain)
class DeviceUtils; // forward decl for DeviceContext::utils (owned generic vector/transpose subdomain)

// Buffer growth helpers shared by all subdomains. These used to be private
// Device templates; promoting them to free functions lets DeviceJk/DevicePdft/
// ... grow their own buffers without reaching back into the Device class.
template<class T>
T* grow_array(PM_NS::PM * pm, T * &ptr, int current_size, int & max_size,
              std::string name, const char * file, int line)
{
  if(current_size > max_size) {
    max_size = current_size;
    if(ptr) pm->dev_free_async(ptr, name);
    ptr = (T *) pm->dev_malloc_async(current_size * sizeof(T), name, file, line);
  }
  return ptr;
}

template<class T>
T* grow_array_host(PM_NS::PM * pm, T * &ptr, int current_size, int & max_size,
                   std::string name)
{
  if(current_size > max_size) {
    max_size = current_size;
    if(ptr) pm->dev_free_host(ptr);
    ptr = (T *) pm->dev_malloc_host(current_size * sizeof(T));
  }
  return ptr;
}

struct DeviceJkData {
  int size_rho, size_vj, size_vk, size_buf1, size_buf2, size_buf3;
  int size_dms, size_dmtril;
  double *d_rho, *d_vj, *d_buf1, *d_buf2, *d_buf3, *d_vkk;
  double *d_dms, *d_dmtril;
};

struct DeviceAo2moData {
  int size_j_pc, size_k_cp, size_k_pc, size_bufd, size_bufpa, size_bufaa;
  double *d_j_pc, *d_k_pc, *d_bufd, *d_bufpa, *d_bufaa;
  double *d_ppaa, *d_papa;
};

struct DeviceH2effData {
  int size_eri_unpacked, size_eri_h2eff;
  double *d_eri_h2eff;
};

struct DeviceImphamData {
};

struct DevicePdftData {
  int size_mo_grid, size_ao_grid, size_cascm2, size_Pi, size_buf_pdft;
  double *d_ao_grid, *d_cascm2, *d_mo_grid, *d_Pi, *d_buf_pdft1, *d_buf_pdft2;
};

struct DeviceFciData {
  int size_clinka, size_clinkb, size_cibra, size_ciket;
  int size_tdm1, size_tdm2, size_tdm2_p, size_pdm1, size_pdm2;
  int *d_clinka, *d_clinkb;
  double *d_cibra, *d_ciket;
  double *d_tdm1, *d_tdm2, *d_tdm2_p;
  double *d_tdm1h, *d_tdm3ha, *d_tdm3hb;
  double *d_pdm1, *d_pdm2;
  std::vector<int> type_pumap, size_pumap;
  std::vector<int *> pumap, d_pumap;
  int *d_pumap_ptr; // no explicit allocation
  int *pumap_ptr; // no explicit allocation
};

struct DeviceLassiData {
};

struct my_device_data {
  int device_id;
  int active; // was device used in calculation and has result to be accumulated?

  // Shared
  int size_mo_coeff;
  int size_mo_cas;
  int size_ucas;
  int size_umat;
  int size_h2eff;
  double * d_mo_coeff;
  double * d_mo_cas;
  double * d_ucas;
  double * d_umat;
  double * d_h2eff;

  // Per-domain data
  DeviceJkData jk;
  DeviceAo2moData ao2mo;
  DeviceH2effData h2eff;
  DeviceImphamData impham;
  DevicePdftData pdft;
  DeviceFciData fci;
  DeviceLassiData lassi;

  // we keep the following for now, but we don't explicitly use them anymore
  // besides, pm.h should defined a queue_t and mathlib.h a handle_t...

#if defined (_USE_GPU)
#if defined _GPU_CUBLAS
  cublasHandle_t handle;
  cudaStream_t stream;
#elif defined _GPU_HIPBLAS
  hipblasHandle_t handle;
  hipStream_t stream;
#elif defined _GPU_MKL
  int * handle;
  sycl::queue * stream;
#endif
#else
  int * handle;
  int * stream;
#endif
};

// One profiled accumulation site, registered on first use by DeviceContext::profile.
// Sites are keyed by their identity (class + function, parsed from __PRETTY_FUNCTION__
// at the call site), so the report and the increment sites cannot disagree on which
// timer/counter a method owns.
struct ProfileSite {
  std::string cls;
  std::string name;
  double time;
  size_t count;
};

// "void DeviceFci::compute_sivecs(int)" -> cls="DeviceFci", name="compute_sivecs"
static inline void parse_site(const char * pretty, std::string & cls, std::string & name)
{
  std::string s(pretty);
  size_t paren = s.find('(');
  if(paren != std::string::npos) s = s.substr(0, paren);
  size_t sc = s.rfind("::");
  if(sc == std::string::npos) {
    name = s;
    cls.clear();
  } else {
    name = s.substr(sc + 2);
    size_t sp = s.find(' ');
    cls = (sp == std::string::npos) ? s.substr(0, sc) : s.substr(sp + 1, sc - sp - 1);
  }
}

// Shared resources borrowed by every subdomain (owned by the Device facade).
// Subdomains access PM/MATHLIB/per-device state through this instead of owning
// copies of the pointers.
struct DeviceContext {
  PM_NS::PM * pm;
  MATHLIB_NS::MATHLIB * ml;
  DeviceComm * comm;     // multi-GPU bcast/reduce (owned by the Device facade)
  DeviceCache * cache;   // ERI-block + pumap cache (owned by the Device facade)
  DeviceUtils * utils;   // generic vector/transpose kernels (owned by the Device facade)
  int num_devices;
  int verbose_level;
  int grid_size, block_size;
  my_device_data * device_data;

  // Profiling: ordered by first use (report order), with a name -> index map for
  // O(1) find-or-insert. Every timed site is also counted. NOT mutex-guarded:
  // every LIBGPU_PROFILE call site sits on the host thread after its OpenMP
  // parallel region has joined, and the pybind entry points are GIL-serialized,
  // so profile() is never reached concurrently. If a future call path can run
  // concurrently (async host threads, callbacks), accumulate on a single thread
  // at the call site rather than locking here.
  std::vector<ProfileSite> profile_sites;
  std::unordered_map<std::string, size_t> site_index;

  // Accumulate one profiling sample. Every timed site is also counted. Call through
  // the LIBGPU_PROFILE macro so the call site's identity (class + function via
  // __PRETTY_FUNCTION__) is recorded.
  void profile(double dt, const char * fn)
  {
    std::string cls, name;
    parse_site(fn, cls, name);
    std::string key = cls + "::" + name;
    size_t idx;
    auto it = site_index.find(key);
    if(it == site_index.end()) {
      idx = profile_sites.size();
      profile_sites.push_back({cls, name, 0.0, 0});
      site_index[key] = idx;
    } else {
      idx = it->second;
    }
    profile_sites[idx].time += dt;
    profile_sites[idx].count += 1;
  }
};

// Accumulate one profiling sample at the call site, recording the enclosing
// function's identity (class + method via __PRETTY_FUNCTION__) in ctx.profile_sites.
#define LIBGPU_PROFILE(ctx, dt) \
  (ctx).profile((dt), __PRETTY_FUNCTION__)

#endif
