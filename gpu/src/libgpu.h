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
  
  void libgpu_set_verbose_(void *, int);
  
  int libgpu_get_num_devices(void *);
  void libgpu_dev_properties(void *, int);
  void libgpu_set_device(void *, int);

  void libgpu_disable_eri_cache_(void *);
  
  void libgpu_init_get_jk(void *,
			  py::array_t<double>, py::array_t<double>, int, int, int, int, int);

  void libgpu_compute_get_jk(void *,
			     int, int, int,
			     py::array_t<double>, py::array_t<double>, py::list &,
			     py::array_t<double>, py::array_t<double>,
			     int, int, size_t);
  
  void libgpu_pull_get_jk(void *, py::array_t<double>, py::array_t<double>, int, int, int);
  void libgpu_set_update_dfobj_(void *, int);
  void libgpu_get_dfobj_status(void *, size_t, py::array_t<int>);
  
  void libgpu_push_mo_coeff(void *, 
			    py::array_t<double>, int);
  void libgpu_extract_mo_cas(void *, 
			    int, int, int);
  
  void libgpu_init_jk_ao2mo(void *, 
                            int, int);
  void libgpu_init_ints_ao2mo(void *, 
                            int, int, int);
  void libgpu_init_ints_ao2mo_v3(void *, 
                            int, int, int);
  void libgpu_init_ppaa_ao2mo(void *, 
                             int, int);
  void libgpu_df_ao2mo_pass1_v2(void * ,
                             int, int, int, int, int, int,
                             py::array_t<double>, int, size_t);
  void libgpu_df_ao2mo_v3(void * ,
                             int, int, int, int, int, int,
                             py::array_t<double>, int, size_t);
  void libgpu_pull_jk_ao2mo(void *, 
                            py::array_t<double>, py::array_t<double>,int, int);
  void libgpu_pull_ints_ao2mo(void *, 
			      py::array_t<double>, py::array_t<double>, int, int, int, int);
  void libgpu_pull_ints_ao2mo_v3(void *, 
			      py::array_t<double>, int, int, int, int);
  void libgpu_pull_ppaa_ao2mo(void *, 
			      py::array_t<double>, int, int);

  void libgpu_orbital_response(void *,
			       py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       int, int, int); 
  void libgpu_update_h2eff_sub(void *, 
                               int, int, int, int, 
			       py::array_t<double>, py::array_t<double>);
  void libgpu_init_eri_h2eff(void * , 
                              int, int);
  void libgpu_get_h2eff_df(void * , 
                           py::array_t<double> , 
                           int , int , int , int , int ,
                           py::array_t<double>, int, size_t);
  void libgpu_get_h2eff_df_v1(void * , 
                           py::array_t<double> , 
                           int , int , int , int , int ,
                           py::array_t<double>, int, size_t);
  void libgpu_get_h2eff_df_v2(void * , 
                           py::array_t<double> , 
                           int , int , int , int , int ,
                           py::array_t<double>, int, size_t);
  void libgpu_pull_eri_h2eff(void * , 
                              py::array_t<double>, int, int);
}


PYBIND11_MODULE(libgpu, m) {
  m.doc() = "gpu accelerator library example";

  m.def("libgpu_init", &libgpu_init, py::return_value_policy::reference, "simple initialization of device");
  m.def("libgpu_create_device", &libgpu_create_device, py::return_value_policy::reference, "create Device object");
  m.def("libgpu_destroy_device", &libgpu_destroy_device, "destroy Device object");
  
  m.def("libgpu_set_verbose_", &libgpu_set_verbose_, "set verbosity level");
  m.def("libgpu_get_num_devices", &libgpu_get_num_devices, "return number of devices present");
  m.def("libgpu_dev_properties", &libgpu_dev_properties, "info on available devices");
  m.def("libgpu_set_device", &libgpu_set_device, "select device");
  m.def("libgpu_disable_eri_cache_", &libgpu_disable_eri_cache_, "disable caching eri blocks to reduce memory usage for get_jk");

  m.def("libgpu_compute_get_jk", &libgpu_compute_get_jk, "pyscf/df/df_jk.py::get_jk()");
  m.def("libgpu_init_get_jk", &libgpu_init_get_jk, "alloc for get_jk()");
  m.def("libgpu_pull_get_jk", &libgpu_pull_get_jk, "retrieve vj & vk from get_jk()");
  
  m.def("libgpu_set_update_dfobj_", &libgpu_set_update_dfobj_, "ensure that eri is updated on device for get_jk");
  m.def("libgpu_get_dfobj_status", &libgpu_get_dfobj_status, "retrieve info on dfobj and cached eri blocks on device");

  m.def("libgpu_push_mo_coeff", &libgpu_push_mo_coeff, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.1");
  m.def("libgpu_extract_mo_cas", &libgpu_extract_mo_cas, "pyscf/mcscf/las_ao2mo.py");
  m.def("libgpu_init_jk_ao2mo", &libgpu_init_jk_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.2");
  m.def("libgpu_init_ints_ao2mo", &libgpu_init_ints_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.3");
  m.def("libgpu_init_ppaa_ao2mo", &libgpu_init_ppaa_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.4");
  m.def("libgpu_init_ints_ao2mo_v3", &libgpu_init_ints_ao2mo_v3, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.3_v3");
  m.def("libgpu_df_ao2mo_pass1_v2", &libgpu_df_ao2mo_pass1_v2, "pyscf/mcscf/df.py::_ERIS.__init__() 2.0");
  m.def("libgpu_df_ao2mo_v3", &libgpu_df_ao2mo_v3, "pyscf/mcscf/df.py::_ERIS.__init__() 3.0");
  m.def("libgpu_pull_jk_ao2mo", &libgpu_pull_jk_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.5");
  m.def("libgpu_pull_ints_ao2mo", &libgpu_pull_ints_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.6");
  m.def("libgpu_pull_ppaa_ao2mo", &libgpu_pull_ppaa_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.7");
  m.def("libgpu_pull_ints_ao2mo_v3", &libgpu_pull_ints_ao2mo_v3, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.6_v3");
  m.def("libgpu_update_h2eff_sub", &libgpu_update_h2eff_sub, "my_pyscf/mcscf/lasci_sync.py::_update_h2_eff()");
  m.def("libgpu_init_eri_h2eff", &libgpu_init_eri_h2eff, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df part 0.1");
  m.def("libgpu_get_h2eff_df", &libgpu_get_h2eff_df, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df");
  m.def("libgpu_get_h2eff_df_v1", &libgpu_get_h2eff_df_v1, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df_v1");
  m.def("libgpu_get_h2eff_df_v2", &libgpu_get_h2eff_df_v2, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df_v2");
  m.def("libgpu_pull_eri_h2eff", &libgpu_pull_eri_h2eff, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df part 0.3");
  
  m.def("libgpu_orbital_response", &libgpu_orbital_response, "mrh/lasscf_sync_o0.py::orbital_response");
}

#endif
