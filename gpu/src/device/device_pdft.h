/* -*- c++ -*- */

#ifndef DEVICE_PDFT_H
#define DEVICE_PDFT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// MC-PDFT on-top pair density orchestration.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. Per-device buffers live in
// my_device_data::pdft (DevicePdftData). The kernel launchers declared here
// are implemented per platform in pm/<platform>/pdft.cpp.
class DevicePdft {

public:

  DevicePdft(DeviceContext & ctx);
  ~DevicePdft();

  // orchestration (invoked through the Device facade)
  void init_mo_grid(int ngrid, int nmo);
  void push_ao_grid(py::array_t<double> _ao, int ngrid, int nao, int count);
  void compute_mo_grid(int ngrid, int nao, int nmo);
  void pull_mo_grid(py::array_t<double> _mo, int ngrid, int nmo);
  void push_cascm2(py::array_t<double> _cascm2, int ncas);
  void init_Pi(int ngrid);
  void compute_rho_to_Pi(py::array_t<double> _rho, int ngrid, int count);
  void compute_Pi(int ngrid, int ncas, int nao, int count);
  void pull_Pi(py::array_t<double> _Pi, int ngrid, int count);

  // kernel launchers (implemented in pm/<platform>/pdft.cpp)
  void get_rho_to_Pi(double *, double *, int); // replace with gemm or element wise multiplication
  void make_gridkern(double *, double *, int, int); // replace with ml->gemm()
  void make_buf_pdft(double *, double *, double *, int, int); // replace with ml->gemm()
  void make_Pi_final(double *, double *, double *, int, int); // replace with ml->gemm()

private:

  DeviceContext & ctx;
};

#endif
