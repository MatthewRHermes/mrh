/* -*- c++ -*- */

#ifndef DEVICE_JK_H
#define DEVICE_JK_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// J/K Coulomb + exchange build orchestration.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. Per-device buffers live in
// my_device_data::jk (DeviceJkData); the pinned host staging buffers
// buf_vj/buf_vk for the multi-device pull/reduce live here. The kernel
// launchers declared here are implemented per platform in pm/<platform>/jk.cpp.
class DeviceJk {

public:

  DeviceJk(DeviceContext & ctx);
  ~DeviceJk();

  // orchestration (invoked through the Device facade)
  void init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril,
                   int blksize, int nset, int nao, int naux, int count);
  void get_jk(int naux, int nao, int nset,
              py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
              py::array_t<double> _vj, py::array_t<double> _vk,
              int with_k, int count, size_t addr_dfobj);
  void pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk,
                   int nao, int nset, int with_k);

  // kernel launchers (implemented in pm/<platform>/jk.cpp)
  void getjk_rho(double *, double *, double *, int, int, int);
  void getjk_vj(double *, double *, double *, int, int, int, int);

private:

  DeviceContext & ctx;

  double * buf_vj, * buf_vk;
  int size_buf_vj, size_buf_vk;
};

#endif
