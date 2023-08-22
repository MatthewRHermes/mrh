#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "libgpu.h"
#include "pm.h"

// Fortran
//  :: allocate(N, M)
//  :: access (i,j)
// C/C++
//  :: access[(j-1)*N + (i-1)]

#define F_2D(name, i, j, lda) name[(j-1)*lda + i-1]

// might have order of lds flipped...
#define F_3D(name, i, j, k, lda, ldb) name[(k-1)*lda*ldb + (j-1)*lda + i-1]
#define F_4D(name, i, j, k, l, lda, ldb, ldc) name[(l-1)*lda*ldb*ldc + (k-1)*lda*ldb + (j-1)*lda + i-1]

void * libgpu_create_device()
{
  Device * ptr = new Device();
  return (void *) ptr;
}

/* ---------------------------------------------------------------------- */

void libgpu_destroy_device(void * ptr)
{
  Device * dev = (Device *) ptr;
  delete dev;
}

/* ---------------------------------------------------------------------- */

int libgpu_get_num_devices(void * ptr)
{
  Device * dev = (Device *) ptr;
  return dev->get_num_devices();
}

/* ---------------------------------------------------------------------- */

void libgpu_dev_properties(void * ptr, int N)
{
  Device * dev = (Device *) ptr;
  dev->get_dev_properties(N);
}

/* ---------------------------------------------------------------------- */

void libgpu_set_device(void * ptr, int id)
{
  Device * dev = (Device *) ptr;
  dev->set_device(id);
}

/* ---------------------------------------------------------------------- */

void libgpu_setup(void * ptr, py::array_t<double> array, int N)
{
  py::buffer_info info = array.request();
  auto data = static_cast<double*>(info.ptr);
  
  Device * dev = (Device *) ptr;
  dev->setup(data,N);
}

/* ---------------------------------------------------------------------- */

double libgpu_compute(void * ptr, py::array_t<double> array)
{
  py::buffer_info info = array.request();
  auto data = static_cast<double*>(info.ptr);
  
  Device * dev = (Device *) ptr;
  return dev->compute(data);
}

/* ---------------------------------------------------------------------- */

void libgpu_init_get_jk(void * ptr,
			py::array_t<double> eri1, py::array_t<double> dmtril, 
			int blksize, int nset, int nao)
{ 
  Device * dev = (Device *) ptr;
  dev->init_get_jk(eri1, dmtril, blksize, nset, nao);
}

/* ---------------------------------------------------------------------- */

void libgpu_free_get_jk(void * ptr)
{ 
  Device * dev = (Device *) ptr;
  dev->free_get_jk();
}

/* ---------------------------------------------------------------------- */

void libgpu_compute_get_jk(void * ptr,
			   int naux, int nao, int nset,
			   py::array_t<double> eri1, py::array_t<double> dmtril, py::list & dms,
			   py::array_t<double> vj, py::array_t<double> vk,
			   int count)
{ 
  Device * dev = (Device *) ptr;
  dev->get_jk(naux, nao, nset,
	      eri1, dmtril, dms,
	      vj, vk,
	      count);
 
}

/* ---------------------------------------------------------------------- */

void libgpu_orbital_response(void * ptr,
			     py::array_t<double> f1_prime,
			     py::array_t<double> ppaa, py::array_t<double> papa, py::array_t<double> eri_paaa,
			     py::array_t<double> ocm2, py::array_t<double> tcm2, py::array_t<double> gorb,
			     int ncore, int nocc, int nmo)
{
  Device * dev = (Device *) ptr;
  dev->orbital_response(f1_prime,
			ppaa, papa, eri_paaa,
			ocm2, tcm2, gorb,
			ncore, nocc, nmo);
}
