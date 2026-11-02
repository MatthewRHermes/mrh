/* -*- c++ -*- */

#ifndef DEVICE_UTILITIES_H
#define DEVICE_UTILITIES_H

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Generic cross-domain device vector/transpose helpers (shared, owned by none).
//
// Stateless: owns no PM/MATHLIB/device buffers; everything is borrowed through
// the DeviceContext provided by the Device facade. It used to live on Device
// and was reached from subdomains via ctx.owner->vecadd/... ; it is now an
// owned subdomain reachable via ctx.utils->vecadd/...
// The former ctx.owner back-reference no longer exists.
//
// These are the last remaining generic kernels; the domain-specific kernels
// (transpose_120/210, pack_eri) have been migrated to their owning domains
// (DeviceAo2mo, DeviceH2eff, DeviceImpham respectively).
class DeviceUtils {

public:

  DeviceUtils(DeviceContext & ctx) : ctx(ctx) {}

  // kernel launchers (implemented per platform in pm/<platform>/utilities.cpp)
  void transpose(double *, double *, int, int);
  void vecadd(const double *, double *, int); // TODO: replace with ml->daxpy()
  void vecadd_batch(const double *, double *, int, int);
  void memset_zero_batch_stride(double *, int, int, int, int);
  void set_to_zero(double *, int);
  void veccopy(const double *, double *, int);
  void transpose_021(double *, double *, int, int, int);
  void transpose_102(double *, double *, int, int, int);
  void transpose_2130(const double *, double *, int, int, int, int);
  void transpose_3210(double *, double *, int, int);
  void transpose_120(double *, double *, int, int, int, int order = 0);
  void getjk_unpack_buf2(double *, double *, int *, int, int, int);

private:

  DeviceContext & ctx;
};

#endif
