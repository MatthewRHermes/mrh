/* -*- c++ -*- */

#ifndef LIBGPU_H
#define LIBGPU_H

#include "device.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C"
{
  void * libgpu_init();
  void * libgpu_create_device();
  void libgpu_destroy_device(void *);
  
  int libgpu_get_num_devices(void *);
  void libgpu_dev_properties(void *, int);
  void libgpu_set_device(void *, int);

  void libgpu_init_get_jk(void *,
			  py::array_t<double>, py::array_t<double>, int, int, int, int, int);

  void libgpu_compute_get_jk(void *,
			     int,
			     py::array_t<double>, py::array_t<double>, py::list &,
			     py::array_t<double>, py::array_t<double>,
			     int, size_t);
  
  void libgpu_pull_get_jk(void *, py::array_t<double>, py::array_t<double>);
  void libgpu_set_update_dfobj_(void *, int);
  
  void libgpu_hessop_get_veff(void *,
			      int, int, int, int,
			      py::array_t<double>, py::array_t<double>, py::array_t<double>);
  
  void libgpu_hessop_push_bPpj(void *,
			       py::array_t<double>);
  
  void libgpu_orbital_response(void *,
			       py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       int, int, int); 
}


PYBIND11_MODULE(libgpu, m) {
  m.doc() = "gpu accelerator library example";

  m.def("libgpu_init", &libgpu_init, py::return_value_policy::reference, "simple initialization of device");
  m.def("libgpu_create_device", &libgpu_create_device, py::return_value_policy::reference, "create Device object");
  m.def("libgpu_destroy_device", &libgpu_destroy_device, "destroy Device object");
  
  m.def("libgpu_get_num_devices", &libgpu_get_num_devices, "return number of devices present");
  m.def("libgpu_dev_properties", &libgpu_dev_properties, "info on available devices");
  m.def("libgpu_set_device", &libgpu_set_device, "select device");

  m.def("libgpu_compute_get_jk", &libgpu_compute_get_jk, "pyscf/df/df_jk.py::get_jk()");
  m.def("libgpu_init_get_jk", &libgpu_init_get_jk, "alloc for get_jk()");
  m.def("libgpu_pull_get_jk", &libgpu_pull_get_jk, "retrieve vj & vk from get_jk()");
  m.def("libgpu_set_update_dfobj_", &libgpu_set_update_dfobj_, "ensure that eri is updated on device for get_jk");

  m.def("libgpu_hessop_get_veff", &libgpu_hessop_get_veff, "lasci_sync.py::get_veff() for HessianOperator");
  m.def("libgpu_hessop_push_bPpj", &libgpu_hessop_push_bPpj, "bPpj array for HessianOperator");
  
  m.def("libgpu_orbital_response", &libgpu_orbital_response, "mrh/lasscf_sync_o0.py::orbital_response");
}

#endif
