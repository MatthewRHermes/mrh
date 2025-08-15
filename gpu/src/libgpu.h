/* -*- c++ -*- */

#ifndef LIBGPU_H
#define LIBGPU_H

#include "device.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C"
{
  //SETUP
  void * libgpu_init();
  void * libgpu_create_device();
  void libgpu_destroy_device(void *);
  
  void libgpu_set_verbose_(void *, int);
  
  int libgpu_get_num_devices(void *);
  void libgpu_dev_properties(void *, int);
  void libgpu_set_device(void *, int);

  void libgpu_disable_eri_cache_(void *);
  void libgpu_set_update_dfobj_(void *, int);
  void libgpu_get_dfobj_status(void *, size_t, py::array_t<int>);
  
  //JK
  void libgpu_init_get_jk(void *,
			  py::array_t<double>, py::array_t<double>, int, int, int, int, int);

  void libgpu_compute_get_jk(void *,
			     int, int, int,
			     py::array_t<double>, py::array_t<double>, py::list &,
			     py::array_t<double>, py::array_t<double>,
			     int, int, size_t);
  
  void libgpu_pull_get_jk(void *, py::array_t<double>, py::array_t<double>, int, int, int);
  
  //AO2MO
  void libgpu_push_mo_coeff(void *, 
			    py::array_t<double>, int);
  void libgpu_extract_mo_cas(void *, 
			    int, int, int);
  
  void libgpu_init_jk_ao2mo(void *, 
                            int, int);
  void libgpu_init_ppaa_papa_ao2mo(void *, 
                             int, int);
  void libgpu_df_ao2mo_v4(void * ,
                             int, int, int, int, int, int,
                             int, size_t);
  void libgpu_pull_jk_ao2mo_v4(void *, 
                            py::array_t<double>, py::array_t<double>,int, int);
  void libgpu_pull_ppaa_papa_ao2mo_v4(void *, 
			      py::array_t<double>,py::array_t<double>, int, int);
  //ORBITAL RESPONSE
  void libgpu_orbital_response(void *,
			       py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       py::array_t<double>, py::array_t<double>, py::array_t<double>,
			       int, int, int); 
  //UPDATE H2EFF
  void libgpu_update_h2eff_sub(void *, 
                               int, int, int, int, 
			       py::array_t<double>, py::array_t<double>);
  //LAS_AO2MO
  void libgpu_init_eri_h2eff(void * , 
                              int, int);
  void libgpu_get_h2eff_df_v2(void * , 
                           py::array_t<double> , 
                           int , int , int , int , int ,
                           py::array_t<double>, int, size_t);
  void libgpu_pull_eri_h2eff(void * , 
                              py::array_t<double>, int, int);

  //ERI for IMPURINTY HAMILTONIAN
  void libgpu_init_eri_impham(void * ptr, 
                                int, int, int);
  void libgpu_compute_eri_impham(void * , 
                                int, int, int, int, int, size_t, int);
  void libgpu_pull_eri_impham(void * , 
                                py::array_t<double>, int, int, int);
  void libgpu_compute_eri_impham_v2(void * , 
                                int, int, int, int, int, size_t, size_t);
  //PDFT
  void libgpu_init_mo_grid(void *, 
                             int, int); //probably don't need it
  void libgpu_push_ao_grid(void *, 
                             py::array_t<double>, int, int, int);
  void libgpu_compute_mo_grid(void *, 
                              int, int, int);//probably don't need it
  void libgpu_pull_mo_grid(void *, 
                             py::array_t<double>, int, int);//probably don't need it
  void libgpu_init_Pi(void *,  
                       int);
  void libgpu_push_cascm2 (void *,
                   py::array_t<double>, int); 
  void libgpu_compute_rho_to_Pi (void *,
                   py::array_t<double>, int, int); 
  void libgpu_compute_Pi (void *,
                   int, int, int, int); 
  void libgpu_pull_Pi (void *,
                   py::array_t<double>, int, int); 
  //FCI
  void libgpu_init_rdm1(void *,
                       int);
  void libgpu_init_rdm2(void *,
                       int);
  void libgpu_push_ci(void *, 
                      py::array_t<double>,  py::array_t<double>, 
                      int , int);
  void libgpu_push_link_indexa(void *, 
                              int, int , py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void libgpu_push_link_indexb(void *, 
                              int, int , py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void libgpu_push_link_index_ab(void *, 
                              int, int ,int, int, py::array_t<int>, py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void libgpu_compute_trans_rdm1a(void *, 
                            int , int , int , int , int );
  void libgpu_compute_trans_rdm1b(void *, 
                            int , int , int , int , int );
  void libgpu_compute_tdm12kern_a(void *, 
                            int , int , int , int , int );
  void libgpu_compute_tdm12kern_b(void *, 
                            int , int , int , int , int );
  void libgpu_compute_tdm12kern_ab(void *, 
                            int , int , int , int , int );
  void libgpu_pull_rdm1(void *, 
                      py::array_t<double> , int );
  void libgpu_pull_rdm2(void *, 
                      py::array_t<double> , int );
}


PYBIND11_MODULE(libgpu, m) {
  m.doc() = "gpu accelerator library example";

  m.def("init", &libgpu_init, py::return_value_policy::reference, "simple initialization of device");
  m.def("create_device", &libgpu_create_device, py::return_value_policy::reference, "create Device object");
  m.def("destroy_device", &libgpu_destroy_device, "destroy Device object");
  
  m.def("set_verbose_", &libgpu_set_verbose_, "set verbosity level");
  m.def("get_num_devices", &libgpu_get_num_devices, "return number of devices present");
  m.def("dev_properties", &libgpu_dev_properties, "info on available devices");
  m.def("set_device", &libgpu_set_device, "select device");
  m.def("disable_eri_cache_", &libgpu_disable_eri_cache_, "disable caching eri blocks to reduce memory usage for get_jk");

  m.def("compute_get_jk", &libgpu_compute_get_jk, "pyscf/df/df_jk.py::get_jk()");
  m.def("init_get_jk", &libgpu_init_get_jk, "alloc for get_jk()");
  m.def("pull_get_jk", &libgpu_pull_get_jk, "retrieve vj & vk from get_jk()");
  
  m.def("set_update_dfobj_", &libgpu_set_update_dfobj_, "ensure that eri is updated on device for get_jk");
  m.def("get_dfobj_status", &libgpu_get_dfobj_status, "retrieve info on dfobj and cached eri blocks on device");

  m.def("push_mo_coeff", &libgpu_push_mo_coeff, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.1");
  m.def("extract_mo_cas", &libgpu_extract_mo_cas, "pyscf/mcscf/las_ao2mo.py");
  m.def("init_jk_ao2mo", &libgpu_init_jk_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.2");
  m.def("init_ppaa_papa_ao2mo", &libgpu_init_ppaa_papa_ao2mo, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.5 ");
  m.def("df_ao2mo_v4", &libgpu_df_ao2mo_v4, "pyscf/mcscf/df.py::_ERIS.__init__() 4.0");
  m.def("pull_jk_ao2mo_v4", &libgpu_pull_jk_ao2mo_v4, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.5 v4");
  m.def("pull_ppaa_papa_ao2mo_v4", &libgpu_pull_ppaa_papa_ao2mo_v4, "pyscf/mcscf/df.py::_ERIS.__init__() part 0.7 v4");
  m.def("update_h2eff_sub", &libgpu_update_h2eff_sub, "my_pyscf/mcscf/lasci_sync.py::_update_h2_eff()");
  m.def("init_eri_h2eff", &libgpu_init_eri_h2eff, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df part 0.1");
  m.def("get_h2eff_df_v2", &libgpu_get_h2eff_df_v2, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df_v2");
  m.def("pull_eri_h2eff", &libgpu_pull_eri_h2eff, "my_pyscf/mcscf/las_ao2mo.py::get_h2eff_df part 0.3");
  m.def("init_eri_impham", &libgpu_init_eri_impham, "my_pyscf/mcscf/lasscf_async/crunch.py::ImpuritySCF._update_impham_1_ part 0.1");
  m.def("compute_eri_impham", &libgpu_compute_eri_impham, "my_pyscf/mcscf/lasscf_async/crunch.py::ImpuritySCF._update_impham_1_ part 0.2");
  m.def("pull_eri_impham", &libgpu_pull_eri_impham, "my_pyscf/mcscf/lasscf_async/crunch.py::ImpuritySCF._update_impham_1_ part 0.3");
  m.def("compute_eri_impham_v2", &libgpu_compute_eri_impham_v2, "my_pyscf/mcscf/lasscf_async/crunch.py::ImpuritySCF._update_impham_1_ part 0.1-0.3");
  
  m.def("init_mo_grid", &libgpu_init_mo_grid, "pyscf/mcpdft/otfnal.py::grid_ao2mo part 0.1");
  m.def("push_ao_grid", &libgpu_push_ao_grid, "pyscf/mcpdft/otfnal.py::grid_ao2mo part 0.2");
  m.def("compute_mo_grid", &libgpu_compute_mo_grid, "pyscf/mcpdft/otfnal.py::grid_ao2mo part 0.3");
  m.def("pull_mo_grid",&libgpu_pull_mo_grid,"pyscf/mcpdft/otfnal.py::grid_ao2mo part 0.4");
  m.def("init_Pi", &libgpu_init_Pi, "pyscf/mcpdft/otfnal.py::energy_ot part 0.1");
  m.def("push_cascm2", &libgpu_push_cascm2, "pyscf/mcpdft/otfnal.py::energy_ot part 0.2");
  m.def("compute_rho_to_Pi", &libgpu_compute_rho_to_Pi, "pyscf/mcpdft/otfnal.py::energy_ot part 0.3");
  m.def("compute_Pi", &libgpu_compute_Pi, "pyscf/mcpdft/otfnal.py::energy_ot part 0.4");
  m.def("pull_Pi", &libgpu_pull_Pi, "pyscf/mcpdft/otfnal.py::energy_ot part final");
  m.def("orbital_response", &libgpu_orbital_response, "mrh/lasscf_sync_o0.py::orbital_response");
  // RDM can be used from previously made JKs
  m.def("init_rdm1",&libgpu_init_rdm1, "pyscf/fci/rdm.py::allocate rdm1 space");
  m.def("init_rdm2",&libgpu_init_rdm2, "pyscf/fci/rdm.py::allocate rdm2 space");
  // Valay: 8/12/2025: This is done as a test to get FCI running. 
  // specifically with several fragment CI problems, I have thoughts on how to optimize this, over several GPUs 
  // Frag 1 cibra is always on gpu 1, Frag 2 cibra on gpu 2 and so on.
  // ciket for each fragment gets pushed to gpu, does calculation with existing cibra and returns the rdm
  // this can be expanded with modulus and or spreading out. 
  // using mgpu_bcast is also needed at some point
  m.def("push_ci",&libgpu_push_ci,"pyscf/fci/rdm.py::make_rdm1_spin1 with FCItrans_rdm1a/b push ci");
  m.def("push_link_indexa",&libgpu_push_link_indexa,"pyscf/fci/:: push link indexa");
  m.def("push_link_indexb",&libgpu_push_link_indexb,"pyscf/fci/:: push link indexb");
  m.def("push_link_index_ab",&libgpu_push_link_index_ab,"pyscf/fci/:: push link index a and b");
  m.def("compute_trans_rdm1a",&libgpu_compute_trans_rdm1a,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItrans_rdm1a");
  m.def("compute_trans_rdm1b",&libgpu_compute_trans_rdm1b,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItrans_rdm1b");
  m.def("compute_tdm12kern_a",&libgpu_compute_tdm12kern_a,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItdm12kern_a");
  m.def("compute_tdm12kern_b",&libgpu_compute_tdm12kern_b,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItdm12kern_b");
  m.def("compute_tdm12kern_ab",&libgpu_compute_tdm12kern_ab,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItdm12kern_ab");
  m.def("pull_rdm1",&libgpu_pull_rdm1,"pyscf/fci/rdm.py::make_rdm1_spin1 pull_rdm1");        
  m.def("pull_rdm2",&libgpu_pull_rdm2,"pyscf/fci/rdm.py::make_rdm1_spin1 pull_rdm2");        
  
}

#endif
