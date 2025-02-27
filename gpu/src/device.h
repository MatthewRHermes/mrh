/* -*- c++ -*- */

#ifndef DEVICE_H
#define DEVICE_H

#include <chrono>
#include <math.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "pm/pm.h"
#include "mathlib/mathlib.h"
#include "pm/dev_array.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

#define _SIZE_GRID 32
#define _SIZE_BLOCK 256

#define _USE_ERI_CACHE
#define _ERI_CACHE_EXTRA 2

//#define _DEBUG_DEVICE
//#define _DEBUG_ERI_CACHE
//#define _DEBUG_H2EFF
//#define _DEBUG_H2EFF2
//#define _DEBUG_H2EFF_DF
//#define _DEBUG_AO2MO

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
  
  int get_num_devices();
  void get_dev_properties(int);
  void set_device(int);
  void disable_eri_cache_();
  void set_verbose_(int);

  void init_get_jk(py::array_t<double>, py::array_t<double>, int, int, int, int, int);
  void get_jk(int, int, int,
	      py::array_t<double>, py::array_t<double>, py::list &,
	      py::array_t<double>, py::array_t<double>,
	      int, int, size_t);
  void pull_get_jk(py::array_t<double>, py::array_t<double>, int, int, int);

  void getjk_rho(double *, double *, double *, int, int, int);
  void getjk_vj(double *, double *, double *, int, int, int, int);
  void getjk_unpack_buf2(double *, double *, int *, int, int, int);
  void transpose(double*, double*, int, int);
  
  void set_update_dfobj_(int);
  void get_dfobj_status(size_t, py::array_t<int>);
 
  void init_jk_ao2mo (int, int);

  void init_ints_ao2mo (int, int, int);
  void init_ints_ao2mo_v3 (int, int, int);
  void init_ppaa_ao2mo (int, int);

 
  void df_ao2mo_pass1_v2 (int, int, int, int, int, int,
			    py::array_t<double>,
			    int, size_t);
  void df_ao2mo_v3 (int, int, int, int, int, int,
			    py::array_t<double>,
			    int, size_t);

  void get_bufpa(const double *, double *, int, int, int, int);
  void get_bufaa(const double *, double *, int, int, int, int);
  void transpose_120(double *, double *, int, int, int, int order = 0);
  void get_bufd(const double *, double *, int, int);
  void pull_jk_ao2mo (py::array_t<double>,py::array_t<double>,int, int);
  void pull_ints_ao2mo (py::array_t<double>,py::array_t<double>, int, int, int, int);
  void pull_ints_ao2mo_v3 (py::array_t<double>, int, int, int, int);
  void pull_ppaa_ao2mo (py::array_t<double>, int, int);
  
  

  void orbital_response(py::array_t<double>,
			py::array_t<double>, py::array_t<double>, py::array_t<double>,
			py::array_t<double>, py::array_t<double>, py::array_t<double>,
			int, int, int);

  void update_h2eff_sub(int, int, int, int,
                        py::array_t<double>,py::array_t<double>);
  
  void extract_submatrix(const double *, double *, int, int, int);
  void unpack_h2eff_2d(double *, double *, int *, int, int, int);
  void transpose_2310(double *, double *, int, int);
  void transpose_3210(double *, double *, int, int);
  void pack_h2eff_2d(double *, double *, int *, int, int, int);
  
  void transpose_210(double *, double *, int, int, int);
  
  void init_eri_h2eff( int, int);//VA: new function
  void get_h2eff_df( py::array_t<double> , 
		     int , int , int , int , int ,
		     py::array_t<double>, int, size_t );
  void get_h2eff_df_v1( py::array_t<double> , 
		     int , int , int , int , int ,
		     py::array_t<double>, int, size_t );
  void get_h2eff_df_v2 ( py::array_t<double>, 
                         int, int, int, int, int, 
                         py::array_t<double>, int, size_t);//VA: new function
  void pull_eri_h2eff(py::array_t<double>, int, int);// VA: new function
  void extract_mo_cas(int, int, int);//TODO: fix the difference - changed slightly
  void get_mo_cas(const double *, double *, int, int, int);
  void pack_d_vuwM(const double *, double *, int *, int, int, int);
  void pack_d_vuwM_add(const double *, double *, int *, int, int, int);
  
  void push_mo_coeff(py::array_t<double>, int);
private:

  class PM * pm;

  class MATHLIB * ml;
  
  double host_compute(double *);
  void get_cores(char *);

  void profile_start(const char *);
  void profile_stop();
  void profile_next(const char *);

  int verbose_level;
  
  size_t grid_size, block_size;
  
  // get_jk

  int update_dfobj;

  //  int nset;

  int size_fdrv;
  int size_buf_vj;
  int size_buf_vk;
  
  // get_jk
  
  double * rho;
  //double * vj;
  double * _vktmp;
  
  double * buf_tmp;
  double * buf3;
  double * buf4;
  double * buf_fdrv;

  double * buf_vj;
  double * buf_vk;
  // ao2mo
  int size_buf_k_pc;
  int size_buf_j_pc;
  int size_fxpp; // remove when ao2mo_v3 is running
  int size_bufpa;
  int size_bufaa;
  int size_k_pc;
  int size_j_pc;
  int size_buf_ppaa;

  double * buf_j_pc; 
  double * buf_k_pc; 
  double * pin_fxpp;//remove when ao2mo_v3 is running
  double * pin_bufpa;
  double * buf_ppaa;

  // h2eff_df
  int size_eri_h2eff;
  int size_buf_eri_h2eff;
  double * buf_eri_h2eff;

  // eri caching on device

  bool use_eri_cache;
  
  std::vector<size_t> eri_list; // addr of dfobj+eri1 for key-value pair
  
  std::vector<int> eri_count; // # times particular cache used
  std::vector<int> eri_update; // # times particular cache updated
  std::vector<int> eri_size; // # size of particular cache

  std::vector<int> eri_num_blocks; // # of eri blocks for each dfobj (i.e. trip-count from `for eri1 in dfobj.loop(blksize)`)
  std::vector<int> eri_extra; // per-block data: {naux, nao_pair}
  std::vector<int> eri_device; // device id holding cache

  std::vector<double *> d_eri_cache; // pointers for device caches
  std::vector<double *> d_eri_host; // values on host for checking if update
  
  struct my_AO2MOEnvs {
    int natm;
    int nbas;
    int *atm;
    int *bas;
    double *env;
    int nao;
    int klsh_start;
    int klsh_count;
    int bra_start;
    int bra_count;
    int ket_start;
    int ket_count;
    int ncomp;
    int *ao_loc;
    double *mo_coeff;
    //        CINTOpt *cintopt;
    //        CVHFOpt *vhfopt;
  };

  struct my_device_data {
    int device_id;
    
    int size_rho;
    int size_vj;
    int size_vk;
    int size_buf;
    int size_dms;
    int size_dmtril;
    int size_eri1;
    int size_ucas;
    int size_umat;
    int size_h2eff;
    int size_mo_coeff; 
    int size_mo_cas; 
    //ao2mo
    int size_j_pc;
    int size_k_cp;
    int size_k_pc;
    int size_bufd;
    int size_bufpa;
    int size_bufaa;
    // eri_h2eff
    int size_eri_unpacked;
    int size_eri_h2eff;

    double * d_rho;
    double * d_vj;
    double * d_buf1;
    double * d_buf2;
    double * d_buf3;
    double * d_vkk;
    double * d_dms;
    double * d_dmtril;
    double * d_eri1;
    double * d_ucas;
    double * d_umat;
    double * d_h2eff;
    double * d_mo_coeff;
    double * d_mo_cas;
    //ao2mo
    double * d_j_pc;
    double * d_k_pc;
    double * d_bufd;
    double * d_bufpa;
    double * d_bufaa;
    double * d_ppaa;
    // eri_h2eff
    double * d_eri_h2eff;

    std::vector<int> type_pumap;
    std::vector<int> size_pumap;
    std::vector<int *> pumap;
    std::vector<int *> d_pumap;
    int * d_pumap_ptr; // no explicit allocation
    int * pumap_ptr; // no explicit allocation

    // we keep the following for now, but we don't explicitly use them anymore
    // besides, pm.h should defined a queue_t and mathlib.h a handle_t...
    
#if defined (_USE_GPU)
#if defined _GPU_CUBLAS
    cublasHandle_t handle;
    cudaStream_t stream;
#elif defined _GPU_HIPBLAS
    hipblasHandle_t handle;
    hipStream_t stream;
#elif defined _GPU_MKL
    int * handle;
    sycl::queue * stream;
#endif
#else
    int * handle;
    int * stream;
#endif
  };

  my_device_data * device_data;
  
  int * dd_fetch_pumap(my_device_data *, int, int);
  double * dd_fetch_eri(my_device_data *, double *, int, int, size_t, int);
  double * dd_fetch_eri_debug(my_device_data *, double *, int, int, size_t, int); // we'll trash this after some time
  
  void fdrv(double *, double *, double *,
	    int, int, int *, int *, int, double *);
  
  void ftrans(int,
	      double *, double *, double *,
	      struct my_AO2MOEnvs *);

  int fmmm(double *, double *, double *,
	   struct my_AO2MOEnvs *, int);
  
  void NPdsymm_triu(int, double *, int);
  void NPdunpack_tril(int, double *, double *, int);
/*--------------------------------------------*/
  double * t_array;
  int * count_array;

  int num_threads;
  int num_devices;
};

#endif
