/* -*- c++ -*- */

#ifndef DEVICE_H2EFF_H
#define DEVICE_H2EFF_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Two-electron effective Hamiltonian (h2eff) construction.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. Per-device buffers live in
// my_device_data::h2eff (DeviceH2effData) and in the shared ucas/umat/h2eff
// fields of my_device_data; the pinned host staging buffer buf_eri_h2eff for
// the multi-device pull/reduce lives here and is freed in the destructor.
// The kernel launchers declared here are implemented per platform in
// pm/<platform>/h2eff.cpp. The ERI cache / pumap services (ctx.cache->
// dd_fetch_eri/dd_fetch_pumap) are reached through ctx.cache.
class DeviceH2eff {

public:

  DeviceH2eff(DeviceContext & ctx);
  ~DeviceH2eff();

  // orchestration (invoked through the Device facade)
  void init_eri_h2eff(int, int);
  void update_h2eff_sub(int, int, int, int,
                        py::array_t<double>, py::array_t<double>);
  void get_h2eff_df_v2(py::array_t<double>,
                       int, int, int, int, int,
                       py::array_t<double>, int, size_t);
  void pull_eri_h2eff(py::array_t<double>, int, int);

  // kernel launchers (implemented in pm/<platform>/h2eff.cpp)
  void extract_submatrix(const double *, double *, int, int, int);
  void unpack_h2eff_2d(double *, double *, int *, int, int, int);
  void transpose_2310(double *, double *, int, int);
  void pack_h2eff_2d(double *, double *, int *, int, int, int);
  void pack_d_vuwM(const double *, double *, int *, int, int, int);
  void pack_d_vuwM_add(const double *, double *, int *, int, int, int);
  void transpose_210(double *, double *, int, int, int);

private:

  DeviceContext & ctx;

  double * buf_eri_h2eff;
  int size_buf_eri_h2eff;
};

#endif
