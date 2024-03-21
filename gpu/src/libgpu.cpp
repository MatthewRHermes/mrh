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

/* ---------------------------------------------------------------------- */

void * libgpu_init()
{
  Device * ptr = (Device *) libgpu_create_device();

  int device_id = 0;
  libgpu_set_device(ptr, device_id);
  
  return (void *) ptr;
}

/* ---------------------------------------------------------------------- */

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

void libgpu_init_get_jk(void * ptr,
			py::array_t<double> eri1, py::array_t<double> dmtril, 
			int blksize, int nset, int nao, int naux, int count)
{ 
  Device * dev = (Device *) ptr;
  dev->init_get_jk(eri1, dmtril, blksize, nset, nao, naux, count);
}

/* ---------------------------------------------------------------------- */

void libgpu_compute_get_jk(void * ptr,
			   int naux,
			   py::array_t<double> eri1, py::array_t<double> dmtril, py::list & dms,
			   py::array_t<double> vj, py::array_t<double> vk,
			   int count, size_t addr_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->get_jk(naux,
	      eri1, dmtril, dms,
	      vj, vk,
	      count, addr_dfobj);
 
}

/* ---------------------------------------------------------------------- */

void libgpu_pull_get_jk(void * ptr, py::array_t<double> vj, py::array_t<double> vk)
{ 
  Device * dev = (Device *) ptr;
  dev->pull_get_jk(vj, vk);
}

/* ---------------------------------------------------------------------- */

void libgpu_set_update_dfobj_(void * ptr, int update_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->set_update_dfobj_(update_dfobj);
}

/* ---------------------------------------------------------------------- */

void libgpu_hessop_get_veff(void * ptr,
			    int naux, int nmo, int ncore, int nocc,
			    py::array_t<double> bPpj, py::array_t<double> vPpj, py::array_t<double> vk_bj)
{ 
  Device * dev = (Device *) ptr;
  dev->hessop_get_veff(naux, nmo, ncore, nocc, bPpj, vPpj, vk_bj);
}

/* ---------------------------------------------------------------------- */

void libgpu_hessop_push_bPpj(void * ptr,
			     py::array_t<double> bPpj)
{ 
  Device * dev = (Device *) ptr;
  dev->hessop_push_bPpj(bPpj);
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
