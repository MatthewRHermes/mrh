/* -*- c++ -*- */

#ifndef DEVICE_IMPHAM_H
#define DEVICE_IMPHAM_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Impurity-Hamiltonian ERI build orchestration.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. The pinned host buffer
// pin_eri_impham used for the multi-device pull lives here. All scratch
// buffers (d_buf1/d_buf2/d_buf3) are borrowed from my_device_data::jk;
// the shared ERI-cache/pumap services (ctx.cache->dd_fetch_eri/dd_fetch_pumap)
// and mgpu_reduce are reached through the Device facade.
class DeviceImpham {

public:

  DeviceImpham(DeviceContext & ctx);
  ~DeviceImpham();

  // orchestration (invoked through the Device facade)
  void init_eri_impham(int naoaux, int nao_f, int return_4c2eeri);
  void compute_eri_impham(int nao_s, int nao_f, int blksize, int naux, int count,
                          size_t addr_dfobj, int return_4c2eeri);
  void compute_eri_impham_v2(int nao_s, int nao_f, int blksize, int naux, int count,
                             size_t addr_dfobj_in, size_t addr_dfobj_out);
  void pull_eri_impham(py::array_t<double> _eri, int naoaux, int nao_f, int return_4c2eeri);

  // kernel launchers (implemented in pm/<platform>/impham.cpp)
  void pack_eri(double *, double *, int *, int, int, int);

private:

  DeviceContext & ctx;

  int size_eri_impham;
  double * pin_eri_impham;
};

#endif
