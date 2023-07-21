/* -*- c++ -*- */

#ifndef DEVICE_H
#define DEVICE_H

#include <chrono>
#include <math.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "pm.h"

#define _SIZE_GRID 32
#define _SIZE_BLOCK 256

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

class Device {
  
public :
  
  Device();
  ~Device();
  
  int get_num_devices();
  void get_dev_properties(int);
  void set_device(int);

  void setup(double *, int);
  double compute(double *);

  void init_get_jk(py::array_t<double>, py::array_t<double>, int, int, int);
  void free_get_jk();
  
  void get_jk(py::array_t<double>, py::array_t<double>, py::array_t<double>,
	      py::array_t<double>,
	      py::list &, py::array_t<double>,
	      int, int, int, int, int, int);
  
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

  // get_jk

  int size_rho;
  int size_vj;
  int size_vk;
  int size_buf;
  int size_fdrv;

  double * rho;
  double * vj;
  double * _vktmp;

  double * buf_tmp;
  double * buf3;
  double * buf4;
  double * buf_fdrv;
  
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
#if 1
  void fdrv(double *, double *, double *,
	    int, int, int *, int *, int, double *);
#else
  void fdrv(double *, double *, double *,
	    int, int, int *, int *, int);
#endif
  
  void ftrans(int,
	      double *, double *, double *,
	      struct my_AO2MOEnvs *);

  int fmmm(double *, double *, double *,
	   struct my_AO2MOEnvs *, int);
  
  void NPdsymm_triu(int, double *, int);
  void NPdunpack_tril(int, double *, double *, int);
    
#ifdef _SIMPLE_TIMER
  double total_t;

  double * t_array;
  double * t_array_jk;
#endif
};

#endif