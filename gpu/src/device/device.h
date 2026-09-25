/* -*- c++ -*- */

#ifndef DEVICE_H
#define DEVICE_H

#include <chrono>
#include <math.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "../pm/dev_array.h"
#include "device_context.h"
#include "device_comm.h"
#include "device_cache.h"
#include "device_utilities.h"
#include "device_pdft.h"
#include "device_jk.h"
#include "device_impham.h"
#include "device_lassi.h"
#include "device_h2eff.h"
#include "device_ao2mo.h"
#include "device_fci.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

#define _SIZE_GRID 32
#define _SIZE_BLOCK 256

#define _ERI_CACHE_EXTRA 2

#define _ENABLE_P2P

//#define _DEBUG_DEVICE
//#define _DEBUG_P2P
//#define _DEBUG_FCI
//#define _TEMP_BUFSIZING
//#define _CUSTOM_FCI

#define _PUMAP_2D_UNPACK 0       // generic unpacking of 1D array to 2D matrix
#define _PUMAP_H2EFF_UNPACK 1    // unpacking h2eff array (generic?)
#define _PUMAP_H2EFF_PACK 2      // unpacking h2eff array (generic?)

#define OUTPUTIJ        1
#define INPUT_IJ        2

// pyscf/pyscf/lib/np_helper/np_helper.h
#define BLOCK_DIM    104

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

#define TRIU_LOOP(I, J) \
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) \
                for (I = 0, j1 = MIN(j0+BLOCK_DIM, n); I < j1; I++) \
                        for (J = MAX(I,j0); J < j1; J++)

extern "C" {
  void dsymm_(const char*, const char*, const int*, const int*,
	      const double*, const double*, const int*,
	      const double*, const int*,
	      const double*, double*, const int*);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
	      const int * k, const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb, const double * beta, double * c,
	      const int * ldc);
}

class Device {
  
public :
  
  Device();
  ~Device();
  
  //SETUP
  int get_num_devices();
  void get_dev_properties(int);
  void set_device(int);
  void barrier_all();
  void set_verbose_(int);

  //JK
  void init_get_jk(py::array_t<double>, py::array_t<double>, int, int, int, int, int);
  void get_jk(int, int, int,
	      py::array_t<double>, py::array_t<double>, py::list &,
	      py::array_t<double>, py::array_t<double>,
	      int, int, size_t);
  void pull_get_jk(py::array_t<double>, py::array_t<double>, int, int, int);

  void set_update_dfobj_(int); // forwarder -> DeviceCache
  void get_dfobj_status(size_t, py::array_t<int>); // forwarder -> DeviceCache
 
  //AO2MO
  void init_jk_ao2mo (int, int);

  void init_ppaa_papa_ao2mo (int, int);
 
  void df_ao2mo_v4 (int, int, int, int, int, int,
			    int, size_t);
  void pull_jk_ao2mo_v4 (py::array_t<double>,py::array_t<double>,int, int);
  void pull_ppaa_papa_ao2mo_v4 (py::array_t<double>,py::array_t<double>, int, int);
  
  // DEPRECATED: legacy integral engine (orbital_response) -- commented out for the
  // time being. See refactor_plan.md. If revived, restore this decl, the impl in
  // device.cpp, the binding in libgpu.h/cpp, and the fdrv helper below.
  //ORBITAL RESPONSE
  //void orbital_response(py::array_t<double>,
  //			py::array_t<double>, py::array_t<double>, py::array_t<double>,
  //			py::array_t<double>, py::array_t<double>, py::array_t<double>,
  //			int, int, int);

  //UPDATE H2EFF
  void update_h2eff_sub(int, int, int, int,
                        py::array_t<double>,py::array_t<double>);
  
  //LAS_AO2MO
  void init_eri_h2eff( int, int);//VA: new function
  void get_h2eff_df_v2 ( py::array_t<double>, 
                         int, int, int, int, int, 
                         py::array_t<double>, int, size_t);//VA: new function
  void pull_eri_h2eff(py::array_t<double>, int, int);// VA: new function
  
  //ERI for IMPURINTY HAMILTONIAN (forwarders -> DeviceImpham)
  void init_eri_impham(int, int, int);
  void compute_eri_impham(int, int, int, int, int, size_t, int);
  void pull_eri_impham( py::array_t<double>, int, int, int);
  void compute_eri_impham_v2(int, int, int, int, int, size_t, size_t);
  
  //PDFT
  void init_mo_grid(int, int);
  void push_ao_grid(py::array_t<double>, int, int, int);
  void compute_mo_grid(int, int, int);
  void pull_mo_grid(py::array_t<double>, int, int);
  void init_Pi(int);
  void push_cascm2 (py::array_t<double>, int); 
  void compute_rho_to_Pi (py::array_t<double>, int, int); 
  void compute_Pi (int, int, int, int); 
  void pull_Pi (py::array_t<double>, int, int); 

  //FCI (tdm/rdm orchestration -> DeviceFci via _fci shims)
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

  void init_ox1_pinned(int);

  void push_op(py::array_t<double>, int, int, int);
  void push_op_4frag(py::array_t<double>, int, int, int);
  void push_d2(py::array_t<double>, int, int, int);
  void push_d3(py::array_t<double>, int, int, int);

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

  //inner functions
  void extract_mo_cas(int, int, int, int);

  void push_mo_coeff(py::array_t<double>, int);

  // generic vector/transpose kernels moved to DeviceUtils (ctx.utils->)

private:

  class PM * pm;

  class MATHLIB * ml;
  
  double host_compute(double *);
  void get_cores(char *);

  int verbose_level;
  
  size_t grid_size, block_size;
  
  // DEPRECATED: legacy integral engine state (only used by orbital_response/fdrv).
  //int size_fdrv;
  
  // get_jk
  
  double * rho;
  //double * vj;
  double * _vktmp;
 
  // DEPRECATED: legacy integral engine state (only used by orbital_response/fdrv).
  //double * buf_fdrv;
  // ao2mo
  int size_fxpp; // remove when ao2mo_v3 is running
  int size_bufpa;
  int size_bufaa;
  int size_k_pc;
  int size_j_pc;

  double * pin_fxpp;//remove when ao2mo_v3 is running
  double * pin_bufpa;

  DeviceAo2mo * _ao2mo; // ao2mo domain; owns buf_j_pc/buf_k_pc/buf_ppaa/buf_papa staging
  DeviceFci * _fci;     // fci domain; owns h_bravecs/h_ketvecs/h_dm1_full/h_dm2_full/h_dm2_p_full staging

  // DEPRECATED: legacy integral engine helpers (fdrv/ftrans/fmmm/my_AO2MOEnvs/
  // NPdsymm_triu/NPdunpack_tril) -- commented out for the time being.
  //struct my_AO2MOEnvs {
  //  int natm;
  //  int nbas;
  //  int *atm;
  //  int *bas;
  //  double *env;
  //  int nao;
  //  int klsh_start;
  //  int klsh_count;
  //  int bra_start;
  //  int bra_count;
  //  int ket_start;
  //  int ket_count;
  //  int ncomp;
  //  int *ao_loc;
  //  double *mo_coeff;
  //  //        CINTOpt *cintopt;
  //  //        CVHFOpt *vhfopt;
  //};
  my_device_data * device_data;
  
  template<class T>
  void grow_array(T * &ptr, int current_size, int & max_size, std::string name, const char * file, int line)
  {
    ::grow_array(pm, ptr, current_size, max_size, name, file, line);
  }
  
  template<class T>
  void grow_array_host(T * &ptr, int current_size, int & max_size, std::string name)
  {
    ::grow_array_host(pm, ptr, current_size, max_size, name);
  }
  
  // DEPRECATED: legacy integral engine helpers (see my_AO2MOEnvs above).
  //void fdrv(double *, double *, double *,
  //	    int, int, int *, int *, int, double *);
  //
  //void ftrans(int,
  //	      double *, double *, double *,
  //	      struct my_AO2MOEnvs *);
  //
  //int fmmm(double *, double *, double *,
  //	   struct my_AO2MOEnvs *, int);
  //
  //void NPdsymm_triu(int, double *, int);
  //void NPdunpack_tril(int, double *, double *, int);
/*--------------------------------------------*/

  int num_threads;
  int num_devices;

  DeviceContext dev_ctx;
  DeviceComm * _comm;
  DeviceCache * _cache;
  DeviceUtils * _utils;
  DevicePdft * _pdft;
  DeviceJk * _jk;
  DeviceImpham * _impham;
  DeviceLassi * _lassi;
  DeviceH2eff * _h2eff;
};

#endif
