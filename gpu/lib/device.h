/* -*- c++ -*- */

#ifndef DEVICE_H
#define DEVICE_H

#include <chrono>
#include <math.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "pm.h"

#define _NUM_ITER 10

#define _SIZE_GRID 32
#define _SIZE_BLOCK 256

#define TOL 1e-6

class Device {
  
public :
  
  Device();
  ~Device();
  
  int get_num_devices();
  void get_dev_properties(int);
  void set_device(int);

  void setup(double *, int);
  double compute(double *);

  void orbital_response(py::array_t<double>,
			py::array_t<double>, py::array_t<double>, py::array_t<double>,
			py::array_t<double>, py::array_t<double>, py::array_t<double>,
			int, int, int);
  
private:
  double host_compute(double *);
  int n;
  int size_data;

  double * d_data;
  size_t grid_size, block_size;
  
  double * partial;
  double * d_partial;

  double total_t;
};

#endif
