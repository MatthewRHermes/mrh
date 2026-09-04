/* -*- c++ -*- */

#ifndef DEVICE_FCI_H
#define DEVICE_FCI_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// FCI reduced density matrix (tdm/rdm) orchestration.
//
// Owns no PM/MATHLIB/device state; everything is borrowed through the
// DeviceContext provided by the Device facade. Per-device FCI buffers live in
// my_device_data::fci (DeviceFciData); the pinned host staging buffers
// h_bravecs/h_ketvecs/h_dm1_full/h_dm2_full/h_dm2_p_full for the multi-device
// pull/reduce live here and are freed in the destructor. The inner kernel
// launchers declared here are implemented per platform in pm/<platform>/fci.cpp.
// Generic vector helpers (set_to_zero, veccopy, transpose_021, ...) are
// reached through ctx.utils (DeviceUtils).
class DeviceFci {

public:

  DeviceFci(DeviceContext & ctx);
  ~DeviceFci();

  // orchestration (invoked through the Device facade)
  void init_tdm1(int);
  void init_tdm2(int);
  void init_tdm3hab(int);
  void init_tdm1_host(int);
  void init_tdm2_host(int);
  void init_tdm3h_host(int);
  void copy_bravecs_host(py::array_t<double>, int , int, int);
  void copy_ketvecs_host(py::array_t<double>, int , int, int);
  void push_cibra_from_host(int, int , int, int);
  void push_ciket_from_host(int, int , int, int);

  void push_cibra(py::array_t<double>, int , int, int);
  void push_ciket(py::array_t<double>, int , int, int);
  void push_link_indexa(int, int , py::array_t<int> );
  void push_link_indexb(int, int , py::array_t<int> );
  void compute_trans_rdm1a(int , int , int , int , int, int );
  void compute_trans_rdm1b(int , int , int , int , int, int );
  void compute_make_rdm1a(int , int , int , int , int, int );
  void compute_make_rdm1b(int , int , int , int , int, int );
  void compute_tdm12kern_a_v2(int , int , int , int , int, int );
  void compute_tdm12kern_b_v2(int , int , int , int , int, int );
  void compute_tdm12kern_ab_v2(int , int , int , int , int, int );
  void compute_rdm12kern_sf_v2(int , int , int , int , int, int );
  void compute_tdm13h_spin_v4( int , int , int , int , int , int, int,
                               int , int , int , int , int ,
                               int , int , int , int , int , int);
  void compute_tdm13h_spin_v5( int , int , int , int , int , int, int,
                               int , int , int , int , int ,
                               int , int , int , int , int , int);
  void compute_tdmpp_spin_v4( int , int , int , int , int , int,
                               int , int , int , int , int ,
                               int , int , int , int , int , int);
  void compute_sfudm_v2( int , int , int , int , int,
                      int , int , int , int , int ,
                      int , int , int , int , int , int);
  void compute_tdm1h_spin( int , int , int , int , int , int,
                           int , int , int , int , int ,
                           int , int , int , int , int , int);

  void reorder_rdm(int, int);
  void transpose_tdm2(int, int);
  void pull_tdm1(py::array_t<double> , int, int );
  void pull_tdm2(py::array_t<double> , int, int );

  void pull_tdm1_host(int, int, int, int, int, int, int);
  void pull_tdm2_host(int, int, int, int, int, int, int);
  void pull_tdm3h_host(int, int, int);
  void pull_tdm3hab(py::array_t<double> ,py::array_t<double> , int, int );
  void pull_tdm3hab_v2(py::array_t<double>, py::array_t<double> ,py::array_t<double> , int, int, int, int );
  void pull_tdm3hab_v2_host(int, int, int, int, int, int, int, int );

  void copy_tdm1_host_to_page(py::array_t<double> , int );
  void copy_tdm2_host_to_page(py::array_t<double> , int );

  // inner kernel launchers (implemented in pm/<platform>/fci.cpp)
  void compute_FCItrans_rdm1a (double *, double *, double *, int, int, int, int, int *);
  void compute_FCItrans_rdm1b (double *, double *, double *, int, int, int, int, int *);
  void compute_FCItrans_rdm1a_v2 (double *, double *, double *,
                                 int, int,
                                 int, int, int, int,
                                 int, int, int, int, int,
                                 int *);
  void compute_FCItrans_rdm1b_v2 (double *, double *, double *,
                                 int, int,
                                 int, int, int, int,
                                 int, int, int, int, int,
                                 int *);
  void compute_FCImake_rdm1a (double *, double *, double *, int, int, int, int, int *);
  void compute_FCImake_rdm1b (double *, double *, double *, int, int, int, int, int *);
  void compute_FCIrdm2_a_t1ci_v2 (double *, double *, int, int, int, int, int, int*);
  void compute_FCIrdm2_b_t1ci_v2 (double *, double *, int, int, int, int, int, int*);
  void compute_FCIrdm3h_a_t1ci_v2 (double *, double *, int, int, int, int,
                                int, int, int, int, int*);
  void compute_FCIrdm3h_b_t1ci_v2 (double *, double *, int, int, int, int, int,
                                int, int, int, int, int*);
  void compute_FCIrdm3h_a_t1ci_v3 (double *, double *, int, int, int, int, int, int,
                                int, int, int, int, int*);
  void compute_FCIrdm3h_b_t1ci_v3 (double *, double *, int, int, int, int, int, int,
                                int, int, int, int, int*);
  void transpose_jikl(double *, double *, int);
  void reduce_buf3_to_rdm(const double *, double *, int, int);
  void reorder(double *, double *, double *, int);
  void filter_sfudm(const double *, double *, int);
  void filter_tdmpp(const double *, double *, int, int);
  void filter_tdm1h(const double *, double *, int);
  void filter_tdm3h(double *, double *, int);

private:

  DeviceContext & ctx;

  double * h_bravecs, * h_ketvecs, * h_dm1_full, * h_dm2_full, * h_dm2_p_full;
  int size_bravecs, size_ketvecs, size_dm1_full, size_dm2_full;
};

#endif
