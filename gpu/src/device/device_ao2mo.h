/* -*- c++ -*- */

#ifndef DEVICE_AO2MO_H
#define DEVICE_AO2MO_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// AO-to-MO integral transformation (ao2mo) orchestration.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. Per-device buffers live in
// my_device_data::ao2mo (DeviceAo2moData); the pinned host staging buffers
// buf_j_pc/buf_k_pc/buf_ppaa/buf_papa for the multi-device pull/reduce live
// here and are freed in the destructor. The kernel launchers declared here
// are implemented per platform in pm/<platform>/ao2mo.cpp. The ERI cache /
// pumap services (ctx.cache->dd_fetch_eri/dd_fetch_pumap), mgpu_reduce and
// getjk_unpack_buf2 are reached through the Device facade.
class DeviceAo2mo {

public:

  DeviceAo2mo(DeviceContext & ctx);
  ~DeviceAo2mo();

  // orchestration (invoked through the Device facade)
  void init_jk_ao2mo(int, int);
  void init_ppaa_papa_ao2mo(int, int);
  void df_ao2mo_v4(int, int, int, int, int, int, int, size_t);
  void pull_jk_ao2mo_v4(py::array_t<double>, py::array_t<double>, int, int);
  void pull_ppaa_papa_ao2mo_v4(py::array_t<double>, py::array_t<double>, int, int);
  void extract_mo_cas(int, int, int, int);

  // kernel launchers (implemented in pm/<platform>/ao2mo.cpp)
  void get_bufpa(const double *, double *, int, int, int, int);
  void get_bufaa(const double *, double *, int, int, int, int);
  void get_bufd(const double *, double *, int, int);
  void get_mo_cas(const double *, double *, int, int, int, int);

private:

  DeviceContext & ctx;

  double * buf_j_pc, * buf_k_pc, * buf_ppaa, * buf_papa;
  int size_buf_j_pc, size_buf_k_pc, size_buf_ppaa, size_buf_papa;
};

#endif
