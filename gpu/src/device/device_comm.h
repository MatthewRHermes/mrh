/* -*- c++ -*- */

#ifndef DEVICE_COMM_H
#define DEVICE_COMM_H

#include <vector>

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Multi-GPU communication (broadcast / binary-tree reduce).
//
// Stateless: owns no PM/MATHLIB/device state; everything is borrowed through
// the DeviceContext provided by the Device facade. It used to live on Device
// and was reached from subdomains via ctx.owner->mgpu_bcast/mgpu_reduce; it is
// now an owned subdomain reachable via ctx.comm->mgpu_bcast/mgpu_reduce.
// The former ctx.owner back-reference no longer exists.
class DeviceComm {

public:

  DeviceComm(DeviceContext & ctx);
  ~DeviceComm();

  void mgpu_bcast(std::vector<double *> d_ptr, double * h_ptr, size_t size);
  void mgpu_reduce(std::vector<double *> d_ptr, double * h_ptr, int N, bool blocking,
                   std::vector<double *> buf_ptr, std::vector<int> active);

private:

  DeviceContext & ctx;
};

#endif
