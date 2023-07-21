/* -*- c++ -*- */

#ifndef LIBGPU_H
#define LIBGPU_H

#include "device.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C"
{
  void * libgpu_create_device();
  void libgpu_destroy_device(void *);
  
  int libgpu_get_num_devices(void *);
  void libgpu_dev_properties(void *, int);
  void libgpu_set_device(void *, int);
  

  void libgpu_setup(void *, py::array_t<double>, int);
  double libgpu_compute(void *, py::array_t<double>);

  void libgpu_init_get_jk(void *,
			  py::array_t<double>, py::array_t<double>, int, int, int);
  
  void libgpu_free_get_jk(void *);
  
  void libgpu_compute_get_jk(void *,
			     py::array_t<double>, py::array_t<double>, py::array_t<double>,
			     py::array_t<double>,
			     py::list &, py::array_t<double>,
			     int, int, int, int, int, int);
  
  void libgpu_orbital_response(void *,
			       py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       int, int, int); 
}


PYBIND11_MODULE(libgpu, m) {
  m.doc() = "gpu accelerator library example";

  m.def("libgpu_create_device", &libgpu_create_device, py::return_value_policy::reference, "create Device object");
  m.def("libgpu_destroy_device", &libgpu_destroy_device, "destroy Device object");
  
  m.def("libgpu_get_num_devices", &libgpu_get_num_devices, "return number of devices present");
  m.def("libgpu_dev_properties", &libgpu_dev_properties, "info on available devices");
  m.def("libgpu_set_device", &libgpu_set_device, "select device");
  
  m.def("libgpu_setup", &libgpu_setup, "setup data structs on device");
  m.def("libgpu_compute", &libgpu_compute, "compute something useful on device");

  m.def("libgpu_compute_get_jk", &libgpu_compute_get_jk, "pyscf/df/df_jk.py::get_jk()");
  m.def("libgpu_init_get_jk", &libgpu_init_get_jk, "alloc for get_jk()");
  m.def("libgpu_free_get_jk", &libgpu_free_get_jk, "free for get_jk()");
  
  m.def("libgpu_orbital_response", &libgpu_orbital_response, "mrh/lasscf_sync_o0.py::orbital_response");
}

#endif