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

  void libgpu_barrier(void *);

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
  void libgpu_init_tdm1(void *,
                       int);
  void libgpu_init_tdm2(void *,
                       int);
  void libgpu_init_tdm3hab(void *,
                       int);
  void libgpu_init_tdm1_host(void *,
                       int);
  void libgpu_init_tdm2_host(void *,
                       int);
  void libgpu_init_tdm3h_host(void *,
                       int);
  void libgpu_copy_bravecs_host(void *, 
                      py::array_t<double>, int , int, int);
  void libgpu_copy_ketvecs_host(void *, 
                      py::array_t<double>, int , int, int);
  void libgpu_push_cibra(void *, 
                      py::array_t<double>, int , int, int);
  void libgpu_push_ciket(void *, 
                      py::array_t<double>, int , int, int);
  void libgpu_push_cibra_from_host(void *, 
                      int, int , int, int);
  void libgpu_push_ciket_from_host(void *, 
                      int, int , int, int);
  void libgpu_push_link_indexa(void *, 
                              int, int , py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void libgpu_push_link_indexb(void *, 
                              int, int , py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void libgpu_push_link_index_ab(void *, 
                              int, int ,int, int, py::array_t<int>, py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version
  void libgpu_compute_trans_rdm1a(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_trans_rdm1b(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_make_rdm1a(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_make_rdm1b(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_tdm12kern_a_v2(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_tdm12kern_b_v2(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_tdm12kern_ab_v2(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_rdm12kern_sf_v2(void *, 
                            int , int , int , int , int , int);
  void libgpu_compute_tdm13h_spin_v4(void *, 
                            int , int , int , int , int , int, int, 
                            int , int , int , int , int ,
                            int , int , int , int , int , int);
  void libgpu_compute_tdm13h_spin_v5(void *, 
                            int , int , int , int , int , int, int, 
                            int , int , int , int , int ,
                            int , int , int , int , int , int);
  void libgpu_compute_tdmpp_spin_v4(void *, 
                            int , int , int , int , int , int,
                            int , int , int , int , int ,
                            int , int , int , int , int , int);
  void libgpu_compute_sfudm_v2(void *, 
                            int , int , int , int , int ,
                            int , int , int , int , int ,
                            int , int , int , int , int , int);
  void libgpu_compute_tdm1h_spin(void *, 
                            int , int , int , int , int , int, 
                            int , int , int , int , int ,
                            int , int , int , int , int , int);
  void libgpu_reorder_rdm(void *, 
                            int, int);
  void libgpu_transpose_tdm2(void *, 
                            int, int);
  void libgpu_pull_tdm1_host(void *, 
                      int, int, int, int, int, int, int);
  void libgpu_pull_tdm2_host(void *, 
                      int, int, int, int, int, int, int);
  void libgpu_pull_tdm3h_host(void *, 
                      int, int, int);
  void libgpu_pull_tdm1(void *, 
                      py::array_t<double> , int , int);
  void libgpu_pull_tdm2(void *, 
                      py::array_t<double> , int , int);
  void libgpu_pull_tdm3hab(void *, 
                      py::array_t<double>, py::array_t<double> , int, int);
  void libgpu_pull_tdm3hab_v2(void *, 
                      py::array_t<double>, py::array_t<double>, py::array_t<double> , int, int, int, int);
  void libgpu_pull_tdm3hab_v2_host(void *, 
                      int, int, int, int, int, int, int, int);
  void libgpu_copy_tdm1_host_to_page(void *, 
                      py::array_t<double> , int );
  void libgpu_copy_tdm2_host_to_page(void *, 
                      py::array_t<double> , int );

  //MATVECS FOR LASSI
  void libgpu_push_op(void *, py::array_t<double>, int, int, int);
  void libgpu_push_op_4frag(void *, py::array_t<double>, int, int, int);
  void libgpu_push_d2(void *, py::array_t<double>, int, int, int);
  void libgpu_push_d3(void *, py::array_t<double>, int, int, int);
  void libgpu_init_ox1_pinned(void *, int);
  void libgpu_init_new_sivecs_host(void * , int, int); 
  void libgpu_init_old_sivecs_host(void *, int, int); 
  void libgpu_push_sivecs_to_host(void * , py::array_t<double>, int, int);
  void libgpu_push_sivecs_to_device(void * , py::array_t<double>, int, int, int);
  void libgpu_bcast_vec(void *, int, int);
  void libgpu_push_instruction_list(void * , py::array_t<int>, int);
  void libgpu_compute_sivecs(void *, int, int, int); 
  void libgpu_compute_sivecs_full(void *, int, int, int, int); 
  void libgpu_compute_sivecs_full_v2(void *, int, int, int, int); 
  void libgpu_compute_sivecs_full_v3(void *, int, int, int, int, int, int, int, int); 
  void libgpu_compute_4frag_matvec(void *, int, int, int, int,
                                           int, int, int, int, 
                                           int, 
                                           int, int, 
                                           int, int, int, int);
  void libgpu_print_sivecs(void *, int, int); 
  void libgpu_add_ox1_pinned(void *, py::array_t<double>, int);
  void libgpu_finalize_ox1_pinned(void *, py::array_t<double>, int);
  void libgpu_pull_sivecs_from_pinned(void *, py::array_t<double>, int, int, int);
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
  m.def("barrier", &libgpu_barrier, "wait for all GPUs to complete queued work");
  
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
  m.def("init_tdm1",&libgpu_init_tdm1, "pyscf/fci/rdm.py::allocate rdm1 space");
  m.def("init_tdm2",&libgpu_init_tdm2, "pyscf/fci/rdm.py::allocate rdm2 space");
  m.def("init_tdm3hab",&libgpu_init_tdm3hab, "mrh/my_pyscf/fci/rdm.py::allocate rdm3hab space");
  m.def("init_tdm1_host",&libgpu_init_tdm1_host, "pyscf/fci/rdm.py::allocate full dm1 space");
  m.def("init_tdm2_host",&libgpu_init_tdm2_host, "pyscf/fci/rdm.py::allocate full dm2 space");
  m.def("init_tdm3h_host",&libgpu_init_tdm3h_host, "my_pyscf/fci/rdm.py::allocate full dm2/2_p space");
  m.def("init_tdm2",&libgpu_init_tdm2, "pyscf/fci/rdm.py::allocate rdm2 space");
  m.def("init_tdm3hab",&libgpu_init_tdm3hab, "mrh/my_pyscf/fci/rdm.py::allocate rdm3hab space");
  m.def("push_cibra",&libgpu_push_cibra,"pyscf/fci/rdm.py:: push cibra");
  m.def("push_ciket",&libgpu_push_ciket,"pyscf/fci/rdm.py:: push ciket");
  m.def("copy_bravecs_host",&libgpu_copy_bravecs_host,"pyscf/fci/rdm.py:: copy bravecs to pinned");
  m.def("copy_ketvecs_host",&libgpu_copy_ketvecs_host,"pyscf/fci/rdm.py:: copy ketvecs to pinned");
  m.def("push_cibra_from_host",&libgpu_push_cibra_from_host,"pyscf/fci/rdm.py:: push cibra from pinned host");
  m.def("push_ciket_from_host",&libgpu_push_ciket_from_host,"pyscf/fci/rdm.py:: push ciket from pinned host");
  m.def("push_link_indexa",&libgpu_push_link_indexa,"pyscf/fci/:: push link indexa");
  m.def("push_link_indexb",&libgpu_push_link_indexb,"pyscf/fci/:: push link indexb");
  m.def("push_link_index_ab",&libgpu_push_link_index_ab,"pyscf/fci/:: push link index a and b");
  m.def("compute_trans_rdm1a",&libgpu_compute_trans_rdm1a,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItrans_rdm1a");
  m.def("compute_trans_rdm1b",&libgpu_compute_trans_rdm1b,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItrans_rdm1b");
  m.def("compute_make_rdm1a",&libgpu_compute_make_rdm1a,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCImake_rdm1a");
  m.def("compute_make_rdm1b",&libgpu_compute_make_rdm1b,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCImake_rdm1b");
  m.def("compute_trans_rdm1b",&libgpu_compute_trans_rdm1b,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItrans_rdm1b");
  m.def("compute_tdm12kern_a_v2",&libgpu_compute_tdm12kern_a_v2,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItdm12kern_a_v2");
  m.def("compute_tdm12kern_b_v2",&libgpu_compute_tdm12kern_b_v2,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItdm12kern_b_v2");
  m.def("compute_tdm12kern_ab_v2",&libgpu_compute_tdm12kern_ab_v2,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCItdm12kern_ab_v2");
  m.def("compute_rdm12kern_sf_v2",&libgpu_compute_rdm12kern_sf_v2,"pyscf/fci/rdm.py::make_rdm1_spin1 compute FCIrdm12kern_sf_v2");
  m.def("compute_tdm13h_spin_v4",&libgpu_compute_tdm13h_spin_v4,"mrh/my_pyscf/fci/rdm.py::trans_rdm13hs compute_v4");
  m.def("compute_tdm13h_spin_v5",&libgpu_compute_tdm13h_spin_v5,"mrh/my_pyscf/fci/rdm.py::trans_rdm13hs compute_v5");
  m.def("compute_tdmpp_spin_v4",&libgpu_compute_tdmpp_spin_v4,"mrh/my_pyscf/fci/rdm.py::trans_rdmhh_v4");
  m.def("compute_sfudm_v2",&libgpu_compute_sfudm_v2,"mrh/my_pyscf/fci/rdm.py::trans_sfudm_v2");
  m.def("compute_tdm1h_spin",&libgpu_compute_tdm1h_spin,"mrh/my_pyscf/fci/rdm.py::trans_tdm1hs");
  m.def("reorder_rdm",&libgpu_reorder_rdm,"pyscf/fci/rdm.py::reorder_rdm");        
  m.def("transpose_tdm2",&libgpu_transpose_tdm2,"pyscf/fci/direct_spin1.py::transpose_tdm2");        
  m.def("pull_tdm1",&libgpu_pull_tdm1,"pyscf/fci/rdm.py::make_rdm12_spin1 pull_tdm1");        
  m.def("pull_tdm2",&libgpu_pull_tdm2,"pyscf/fci/rdm.py::make_rdm12_spin1 pull_tdm2");        
  m.def("pull_tdm1_host",&libgpu_pull_tdm1_host,"my_pyscf/lassi/op_o1/frag.py::pull_tdm1 loop");        
  m.def("pull_tdm2_host",&libgpu_pull_tdm2_host,"my_pyscf/lassi/op_o1/frag.py::pull_tdm2 loop");        
  m.def("pull_tdm3h_host",&libgpu_pull_tdm3h_host,"my_pyscf/lassi/op_o1/frag.py::pull_tdm3h loop");        
  m.def("copy_tdm1_host_to_page",&libgpu_copy_tdm1_host_to_page,"my_pyscf/lassi/op_o1/frag.py:: copy tdm1 to pageable memory");
  m.def("copy_tdm2_host_to_page",&libgpu_copy_tdm2_host_to_page,"my_pyscf/lassi/op_o1/frag.py:: copy tdm2 to pageable memory");
  m.def("pull_tdm3hab",&libgpu_pull_tdm3hab,"mrh/my_pyscf/fci/rdm.py::trans_rdm13hs spin1 pull_tdm13hab");        
  m.def("pull_tdm3hab_v2",&libgpu_pull_tdm3hab_v2,"mrh/my_pyscf/fci/rdm.py::trans_rdm13hs spin1 pull_tdm13hab v2");        
  m.def("pull_tdm3hab_v2_host",&libgpu_pull_tdm3hab_v2_host,"mrh/my_pyscf/fci/rdm.py::trans_rdm13hs spin1 pull_tdm13hab v2 to pinned");        

  m.def("push_op",&libgpu_push_op,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_op to gpu");        
  m.def("push_op_4frag",&libgpu_push_op_4frag,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_op_frag to gpu");        
  m.def("push_d2",&libgpu_push_d2,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_op_d2 to gpu");        
  m.def("push_d3",&libgpu_push_d3,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_op_d3 to gpu");        
  m.def("init_ox1_pinned",&libgpu_init_ox1_pinned,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ for storing results"); 
  m.def("init_new_sivecs_host",&libgpu_init_new_sivecs_host,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ init_new_sivecs_host for storing results");        
  m.def("init_old_sivecs_host",&libgpu_init_old_sivecs_host,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ init_old_sivecs_host for storing inputs");        
  m.def("push_sivecs_to_host",&libgpu_push_sivecs_to_host,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_sivecs from python to pinned");        
  m.def("push_sivecs_to_device",&libgpu_push_sivecs_to_device,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_sivecs from python to gpu");        
  m.def("bcast_vec",&libgpu_bcast_vec,"mrh/my_pyscf/lassi/op_o1/hsi.py::bcast_vec");
  m.def("push_instruction_list",&libgpu_push_instruction_list,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ push_instruction_list to host");        
  m.def("compute_sivecs",&libgpu_compute_sivecs,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ compute_sivecs");  
  m.def("compute_sivecs_full",&libgpu_compute_sivecs_full,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ compute_sivecs");  
  m.def("compute_sivecs_full_v2",&libgpu_compute_sivecs_full_v2,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ compute_sivecs_v2");  
  m.def("compute_sivecs_full_v3",&libgpu_compute_sivecs_full_v3,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ compute_sivecs_v3");  
  m.def("compute_4frag_matvec",&libgpu_compute_4frag_matvec,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ 4frag op X");  
  m.def("print_sivecs",&libgpu_print_sivecs,"mrh/my_pyscf/lassi/op_o1/hsi.py::print_sivecs");  
  m.def("pull_sivecs_from_pinned",&libgpu_pull_sivecs_from_pinned,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ pull_sivecs_from_pinned to pageable");        
  m.def("add_ox1_pinned",&libgpu_add_ox1_pinned,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ add_ox1_from_pinned to pageable");        
  m.def("finalize_ox1_pinned",&libgpu_finalize_ox1_pinned,"mrh/my_pyscf/lassi/op_o1/hsi.py::_opuniq_x_ add_ox1_from_pinned to pageable");        

}

#endif
