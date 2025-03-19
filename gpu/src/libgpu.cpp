#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "libgpu.h"
#include "pm/pm.h"

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

void libgpu_set_verbose_(void * ptr, int verbose)
{ 
  Device * dev = (Device *) ptr;
  dev->set_verbose_(verbose);
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

void libgpu_disable_eri_cache_(void * ptr)
{ 
  Device * dev = (Device *) ptr;
  dev->disable_eri_cache_();
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
			   int naux, int nao, int nset,
			   py::array_t<double> eri1, py::array_t<double> dmtril, py::list & dms,
			   py::array_t<double> vj, py::array_t<double> vk,
			   int with_k, int count, size_t addr_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->get_jk(naux, nao, nset,
	      eri1, dmtril, dms,
	      vj, vk,
	      with_k, count, addr_dfobj);
 
}

/* ---------------------------------------------------------------------- */

void libgpu_pull_get_jk(void * ptr, py::array_t<double> vj, py::array_t<double> vk, int nao, int nset, int with_k)
{ 
  Device * dev = (Device *) ptr;
  dev->pull_get_jk(vj, vk, nao, nset, with_k);
}

/* ---------------------------------------------------------------------- */

void libgpu_set_update_dfobj_(void * ptr, int update_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->set_update_dfobj_(update_dfobj);
}

/* ---------------------------------------------------------------------- */

void libgpu_get_dfobj_status(void * ptr, size_t addr_dfobj, py::array_t<int> arg)
{ 
  Device * dev = (Device *) ptr;
  dev->get_dfobj_status(addr_dfobj, arg);
}

/* ---------------------------------------------------------------------- */

void libgpu_push_mo_coeff(void * ptr,
			  py::array_t<double> mo_coeff, int size_mo_coeff)
{
  Device * dev = (Device *) ptr;
  dev->push_mo_coeff(mo_coeff, size_mo_coeff);
}

/* ---------------------------------------------------------------------- */
void libgpu_extract_mo_cas(void * ptr,
			   int ncas, int ncore, int nao)
{
  Device * dev = (Device *) ptr;
  dev->extract_mo_cas(ncas, ncore, nao);
}

/* ---------------------------------------------------------------------- */


void libgpu_init_jk_ao2mo(void * ptr, 
                          int ncore, int nmo)
{
  Device * dev = (Device *) ptr;
  dev->init_jk_ao2mo(ncore, nmo);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_ints_ao2mo(void * ptr, 
                          int naoaux, int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->init_ints_ao2mo(naoaux, nmo, ncas);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_ints_ao2mo_v3(void * ptr, 
                          int naoaux, int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->init_ints_ao2mo_v3(naoaux, nmo, ncas);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_ppaa_ao2mo(void * ptr, 
                           int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->init_ppaa_ao2mo(nmo, ncas);
}
/* ---------------------------------------------------------------------- */


void libgpu_df_ao2mo_pass1_v2(void * ptr,
				int blksize, int nmo, int nao, int ncore, int ncas, int naux,
				py::array_t<double> eri1,
				int count, size_t addr_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->df_ao2mo_pass1_v2(blksize, nmo, nao, ncore, ncas, naux, eri1, count, addr_dfobj);
}
/* ---------------------------------------------------------------------- */
void libgpu_df_ao2mo_v3(void * ptr,
				int blksize, int nmo, int nao, int ncore, int ncas, int naux,
				py::array_t<double> eri1,
				int count, size_t addr_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->df_ao2mo_v3(blksize, nmo, nao, ncore, ncas, naux, eri1, count, addr_dfobj);
}
/* ---------------------------------------------------------------------- */

void libgpu_pull_jk_ao2mo(void * ptr, 
                          py::array_t<double> j_pc, py::array_t<double> k_pc, int nmo, int ncore)
{
  Device * dev = (Device *) ptr;
  dev->pull_jk_ao2mo(j_pc, k_pc, nmo, ncore);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_ints_ao2mo(void * ptr, 
			    py::array_t<double> fxpp, py::array_t<double> bufpa, int blksize, int naoaux, int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->pull_ints_ao2mo(fxpp, bufpa, blksize, naoaux, nmo, ncas);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_ints_ao2mo_v3(void * ptr, 
			    py::array_t<double> bufpa, int blksize, int naoaux, int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->pull_ints_ao2mo_v3(bufpa, blksize, naoaux, nmo, ncas);
}


/* ---------------------------------------------------------------------- */
void libgpu_pull_ppaa_ao2mo(void * ptr, 
                            py::array_t<double> ppaa, int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->pull_ppaa_ao2mo(ppaa, nmo, ncas);
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

/* ---------------------------------------------------------------------- */

void libgpu_update_h2eff_sub(void * ptr, 
                             int ncore, int ncas, int nocc, int nmo, 
			     py::array_t<double> h2eff_sub, py::array_t<double> umat)
{
  Device * dev = (Device *) ptr;
  dev->update_h2eff_sub(ncore,ncas,nocc,nmo,h2eff_sub,umat);
}

/* ---------------------------------------------------------------------- */
void libgpu_init_eri_h2eff(void * ptr, 
                             int nao, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->init_eri_h2eff(nao, ncas);
}
/* ---------------------------------------------------------------------- */

void libgpu_get_h2eff_df(void * ptr, 
                           py::array_t<double> cderi, 
                           int nao, int nmo, int ncas, int naux, int ncore,
                           py::array_t<double> eri1, int count, size_t addr_dfobj)
{
  Device * dev = (Device *) ptr;
  dev->get_h2eff_df(cderi, nao, nmo, ncas, naux, ncore, eri1, count, addr_dfobj); 
}
/* ---------------------------------------------------------------------- */

void libgpu_get_h2eff_df_v1(void * ptr, 
                           py::array_t<double> cderi, 
                           int nao, int nmo, int ncas, int naux, int ncore,
                           py::array_t<double> eri1, int count, size_t addr_dfobj)
{
  Device * dev = (Device *) ptr;
  dev->get_h2eff_df_v1(cderi, nao, nmo, ncas, naux, ncore, eri1, count, addr_dfobj); 
}
/* ---------------------------------------------------------------------- */
void libgpu_get_h2eff_df_v2(void * ptr, 
                           py::array_t<double> cderi, 
                           int nao, int nmo, int ncas, int naux, int ncore,
                           py::array_t<double> eri1, int count, size_t addr_dfobj)
{
  Device * dev = (Device *) ptr;
  dev->get_h2eff_df_v2(cderi, nao, nmo, ncas, naux, ncore, eri1, count, addr_dfobj); 
}
/* ---------------------------------------------------------------------- */

void libgpu_pull_eri_h2eff(void * ptr, 
                             py::array_t<double> eri, int nao, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->pull_eri_h2eff(eri, nao, ncas);
}
/* ---------------------------------------------------------------------- */
