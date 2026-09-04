/* -*- c++ -*- */

#ifndef DEVICE_LASSI_H
#define DEVICE_LASSI_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// LASSI sparse-operator matvec orchestration.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. The pinned host buffers for
// the si-vectors and the on-site operator (h_new_sivecs / h_old_sivecs /
// h_ox1 / h_instruction_list) live here and are freed in the destructor.
// All device scratch (d_buf1/d_buf2/d_buf3) is borrowed from
// my_device_data::jk; the multi-GPU bcast/reduce and the shared transpose /
// set_to_zero / veccopy utilities are reached through the Device facade.
class DeviceLassi {

public:

  DeviceLassi(DeviceContext & ctx);
  ~DeviceLassi();

  // orchestration (invoked through the Device facade)
  void push_op(py::array_t<double>, int, int, int);
  void push_op_4frag(py::array_t<double>, int, int, int);
  void push_d2(py::array_t<double>, int, int, int);
  void push_d3(py::array_t<double>, int, int, int);
  void init_ox1_pinned(int);
  void init_new_sivecs_host(int, int);
  void init_old_sivecs_host(int, int);
  void push_sivecs_to_host(py::array_t<double>, int, int);
  void push_sivecs_to_device(py::array_t<double>, int, int, int);
  void bcast_vec(int, int);
  void push_instruction_list(py::array_t<int>, int);
  void compute_sivecs(int, int, int);
  void compute_sivecs_full(int, int, int, int);
  void compute_sivecs_full_v2(int, int, int, int);
  void compute_sivecs_full_v3(int, int, int, int, int, int, int, int);
  void compute_4frag_matvec(int, int, int, int,
                            int, int, int, int,
                            int,
                            int, int,
                            int, int, int, int, int);
  void print_sivecs(int, int);
  void pull_sivecs_from_pinned(py::array_t<double>, int, int, int);
  void add_ox1_pinned(py::array_t<double>, int);
  void finalize_ox1_pinned(py::array_t<double>, int);

private:

  DeviceContext & ctx;

  int size_new_sivecs;
  int size_old_sivecs;
  int size_ox1;
  int size_op;
  int size_instruction_list;
  int ox1_on_gpu;
  double * h_new_sivecs;
  double * h_old_sivecs;
  double * h_ox1;
  int * h_instruction_list;
};

#endif
