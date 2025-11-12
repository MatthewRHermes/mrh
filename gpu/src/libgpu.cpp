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

void libgpu_barrier(void *ptr)
{
  Device * dev = (Device *) ptr;
  dev->barrier_all();
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
void libgpu_init_ppaa_papa_ao2mo(void * ptr, 
                           int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->init_ppaa_papa_ao2mo(nmo, ncas);
}
/* ---------------------------------------------------------------------- */
void libgpu_df_ao2mo_v4(void * ptr,
				int blksize, int nmo, int nao, int ncore, int ncas, int naux,
				int count, size_t addr_dfobj)
{ 
  Device * dev = (Device *) ptr;
  dev->df_ao2mo_v4(blksize, nmo, nao, ncore, ncas, naux, count, addr_dfobj);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_jk_ao2mo_v4(void * ptr, 
                          py::array_t<double> j_pc, py::array_t<double> k_pc, int nmo, int ncore)
{
  Device * dev = (Device *) ptr;
  dev->pull_jk_ao2mo_v4(j_pc, k_pc, nmo, ncore);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_ppaa_papa_ao2mo_v4(void * ptr, 
                            py::array_t<double> ppaa,py::array_t<double> papa, int nmo, int ncas)
{
  Device * dev = (Device *) ptr;
  dev->pull_ppaa_papa_ao2mo_v4(ppaa, papa, nmo, ncas);
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
void libgpu_init_eri_impham(void * ptr, 
                               int naoaux, int nao_f, int return_4c2eeri)
{
  Device * dev = (Device *) ptr;
  dev->init_eri_impham(naoaux, nao_f, return_4c2eeri);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_eri_impham(void * ptr, 
                               int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj, int return_4c2eeri)
{
  Device * dev = (Device *) ptr;
  dev->compute_eri_impham(nao_s, nao_f, blksize, naux, count, addr_dfobj, return_4c2eeri);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_eri_impham(void * ptr, 
                               py::array_t<double> _cderi, int naoaux, int nao_f, int return_4c2eeri)
{
  Device * dev = (Device *) ptr;
  dev->pull_eri_impham(_cderi, naoaux, nao_f, return_4c2eeri);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_eri_impham_v2(void * ptr, 
                               int nao_s, int nao_f, int blksize, int naux, int count, size_t addr_dfobj_in, size_t addr_dfobj_out)
{
  Device * dev = (Device *) ptr;
  dev->compute_eri_impham_v2(nao_s, nao_f, blksize, naux, count, addr_dfobj_in, addr_dfobj_out);
}

/* ---------------------------------------------------------------------- */
void libgpu_init_mo_grid(void * ptr, 
                             int ngrid, int nmo)
{
  Device * dev = (Device *) ptr;
  dev->init_mo_grid(ngrid, nmo);
}

/* ---------------------------------------------------------------------- */
void libgpu_push_ao_grid(void * ptr, 
                           py::array_t<double> ao, int ngrid, int nao, int count)
{
  Device * dev = (Device *) ptr;
  dev->push_ao_grid(ao, ngrid, nao, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_compute_mo_grid(void * ptr, 
                            int ngrid, int nao, int nmo)
{
  Device * dev = (Device *) ptr;
  dev->compute_mo_grid(ngrid, nao, nmo);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_mo_grid(void * ptr, 
                          py::array_t<double> mo_grid, int ngrid, int nao)
{
  Device * dev = (Device *) ptr;
  dev->pull_mo_grid(mo_grid, ngrid, nao);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_Pi(void * ptr,  
                     int ngrid)
{
  Device * dev = (Device *) ptr;
  dev->init_Pi(ngrid);
}

/* ---------------------------------------------------------------------- */
void libgpu_push_cascm2 (void * ptr,
                 py::array_t<double> cascm2, int ncas) 
{
  Device * dev = (Device *) ptr;
  dev->push_cascm2(cascm2, ncas);
}

/* ---------------------------------------------------------------------- */
void libgpu_compute_rho_to_Pi(void * ptr, 
                 py::array_t<double> rho, int ngrid, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_rho_to_Pi(rho, ngrid, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_Pi (void * ptr,
                 int ngrid, int ncas, int nao, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_Pi(ngrid, ncas, nao, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_pull_Pi (void * ptr,
                 py::array_t<double> Pi, int ngrid, int count) 
{
  Device * dev = (Device *) ptr;
  dev->pull_Pi(Pi, ngrid, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_init_tdm1(void * ptr, 
                      int norb)
{
  Device * dev = (Device *) ptr;
  dev->init_tdm1(norb);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_tdm1h(void * ptr, 
                      int norb)
{
  Device * dev = (Device *) ptr;
  dev->init_tdm1(norb);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_tdm3hab(void * ptr, 
                      int norb)
{
  Device * dev = (Device *) ptr;
  dev->init_tdm3hab(norb);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_tdm2(void * ptr, 
                      int norb)
{
  Device * dev = (Device *) ptr;
  dev->init_tdm2(norb);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_tdm1_host(void * ptr, int size_dm)
{ 
  Device * dev = (Device *) ptr;
  dev->init_tdm1_host(size_dm);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_tdm2_host(void * ptr, int size_dm)
{ 
  Device * dev = (Device *) ptr;
  dev->init_tdm2_host(size_dm);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_tdm3h_host(void * ptr, int size_dm)
{ 
  Device * dev = (Device *) ptr;
  dev->init_tdm3h_host(size_dm);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_cibra(void * ptr, 
                      py::array_t<double> cibra, int na, int nb, int count)
{
  Device * dev = (Device *) ptr;
  dev->push_cibra(cibra, na, nb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_ciket(void * ptr, 
                      py::array_t<double> ciket, int na, int nb, int count)
{
  Device * dev = (Device *) ptr;
  dev->push_ciket(ciket, na, nb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_copy_bravecs_host(void * ptr, 
                      py::array_t<double> bravecs, int nvecs, int na, int nb)
{
  Device * dev = (Device *) ptr;
  dev->copy_bravecs_host(bravecs, nvecs, na, nb);
}
/* ---------------------------------------------------------------------- */
void libgpu_copy_ketvecs_host(void * ptr, 
                      py::array_t<double> ketvecs, int nvecs, int na, int nb)
{
  Device * dev = (Device *) ptr;
  dev->copy_ketvecs_host(ketvecs, nvecs, na, nb);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_cibra_from_host(void *ptr, 
                              int bra_index, int na, int nb, int count)
{
  Device * dev = (Device *) ptr;
  dev->push_cibra_from_host(bra_index, na, nb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_ciket_from_host(void *ptr, 
                              int ket_index, int na, int nb, int count)
{
  Device * dev = (Device *) ptr;
  dev->push_ciket_from_host(ket_index, na, nb, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_push_link_indexa(void * ptr, 
                              int na, int nlinka, py::array_t<int> link_indexa) //TODO: figure out the shape? or maybe move the compressed version 
{
  Device * dev = (Device *) ptr;
  dev->push_link_indexa(na, nlinka, link_indexa);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_link_indexb(void * ptr, 
                              int nb, int nlinkb, py::array_t<int> link_indexb) //TODO: figure out the shape? or maybe move the compressed version 
{
  Device * dev = (Device *) ptr;
  dev->push_link_indexb(nb, nlinkb, link_indexb);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_link_index_ab(void * ptr, 
                              int na, int nb, int nlinka, int nlinkb, py::array_t<int> link_indexa, py::array_t<int> link_indexb) //TODO: figure out the shape? or maybe move the compressed version 
{
  Device * dev = (Device *) ptr;
  dev->push_link_indexa(na, nlinka, link_indexa);
  dev->push_link_indexb(nb, nlinkb, link_indexb);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_trans_rdm1a(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_trans_rdm1a(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_trans_rdm1b(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_trans_rdm1b(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_make_rdm1a(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_make_rdm1a(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_make_rdm1b(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_make_rdm1b(na, nb, nlinka, nlinkb, norb, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_compute_tdm12kern_a_v2(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_tdm12kern_a_v2(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_tdm12kern_b_v2(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_tdm12kern_b_v2(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_tdm12kern_ab_v2(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_tdm12kern_ab_v2(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_rdm12kern_sf_v2(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->compute_rdm12kern_sf_v2(na, nb, nlinka, nlinkb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_reorder_rdm(void * ptr, 
                         int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->reorder_rdm(norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_transpose_tdm2(void * ptr, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->transpose_tdm2(norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_tdm13h_spin_v4(void * ptr, 
                                 int na, int nb,
                                 int nlinka, int nlinkb, 
                                 int norb, int spin, int _reorder,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
  Device * dev = (Device *) ptr;
  dev->compute_tdm13h_spin_v4(    na, nb,
                                  nlinka,  nlinkb, 
                                  norb,  spin,  _reorder,
                                  ia_bra,  ja_bra,  ib_bra,  jb_bra,  sgn_bra, 
                                  ia_ket,  ja_ket,  ib_ket,  jb_ket,  sgn_ket, count );
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_tdm13h_spin_v5(void * ptr, 
                                 int na, int nb,
                                 int nlinka, int nlinkb, 
                                 int norb, int spin, int _reorder,
                                 int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                                 int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
  Device * dev = (Device *) ptr;
  dev->compute_tdm13h_spin_v5( na, nb,
                               nlinka,  nlinkb, 
                               norb,  spin,  _reorder,
                               ia_bra,  ja_bra,  ib_bra,  jb_bra,  sgn_bra, 
                               ia_ket,  ja_ket,  ib_ket,  jb_ket,  sgn_ket, count );
}

/* ---------------------------------------------------------------------- */
void libgpu_compute_tdmpp_spin_v4(void * ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int spin,
                            int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                            int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
  Device * dev = (Device *) ptr;
  dev->compute_tdmpp_spin_v4(na, nb, nlinka, nlinkb, norb, spin, 
                             ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,      
                             ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count );
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_sfudm_v2( void *ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, 
                            int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                            int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
  Device * dev = (Device *) ptr;
  dev->compute_sfudm_v2(na, nb, nlinka, nlinkb, norb, 
                     ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,      
                     ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count );
}

/* ---------------------------------------------------------------------- */
void libgpu_compute_tdm1h_spin( void *ptr, 
                            int na, int nb, int nlinka, int nlinkb, int norb, int spin,  
                            int ia_bra, int ja_bra, int ib_bra, int jb_bra, int sgn_bra, 
                            int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sgn_ket, int count )
{
  Device * dev = (Device *) ptr;
  dev->compute_tdm1h_spin(na, nb, nlinka, nlinkb, norb, spin, 
                     ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,      
                     ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count );
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm1(void * ptr, 
                      py::array_t<double> tdm1, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm1(tdm1, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm2(void * ptr, 
                      py::array_t<double> tdm2, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm2(tdm2, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm1_host(void * ptr, 
                      int i, int j, int n_bra, int n_ket, int size_dm, int factor, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm1_host(i, j, n_bra, n_ket, size_dm, factor, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm2_host(void * ptr, 
                      int i, int j, int n_bra, int n_ket, int size_dm, int factor, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm2_host(i, j, n_bra, n_ket, size_dm, factor, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm3hab(void * ptr, 
                      py::array_t<double> tdm3ha, py::array_t<double> tdm3hb, int norb, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm3hab(tdm3ha, tdm3hb, norb, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm3hab_v2(void * ptr, 
                      py::array_t<double> tdm1h, py::array_t<double> tdm3ha, py::array_t<double> tdm3hb, int norb, int cre, int spin, int count )
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm3hab_v2(tdm1h, tdm3ha, tdm3hb, norb, cre, spin, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm3hab_v2_host(void * ptr, 
                      int i, int j, int n_bra, int n_ket, int norb, int cre, int spin, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm3hab_v2_host(i, j, n_bra, n_ket, norb, cre, spin, count);
}

/* ---------------------------------------------------------------------- */
void libgpu_pull_tdm3h_host(void * ptr, 
                      int loc, int size_dm, int count)
{
  Device * dev = (Device *) ptr;
  dev->pull_tdm3h_host(loc, size_dm, count);
}
/* ---------------------------------------------------------------------- */
void libgpu_copy_tdm1_host_to_page(void * ptr, 
                          py::array_t<double> dm1_full, int size_dm1_full)
{
  Device * dev = (Device *) ptr;
  dev->copy_tdm1_host_to_page(dm1_full, size_dm1_full);
}
/* ---------------------------------------------------------------------- */
void libgpu_copy_tdm2_host_to_page(void * ptr, 
                          py::array_t<double> dm2_full, int size_dm2_full)
{
  Device * dev = (Device *) ptr;
  dev->copy_tdm2_host_to_page(dm2_full, size_dm2_full);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_op(void * ptr, 
                     py::array_t<double> op, int m, int k)
{
  Device * dev = (Device *) ptr;
  dev->push_op(op, m, k);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_ox1_pinned(void * ptr, 
                               int size)
{
  Device * dev = (Device *) ptr;
  dev->init_ox1_pinned(size);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_new_sivecs_host(void * ptr, 
                                 int m, int n)
{
  Device * dev = (Device *) ptr;
  dev->init_new_sivecs_host(m, n);
}
/* ---------------------------------------------------------------------- */
void libgpu_init_old_sivecs_host(void * ptr, 
                                 int k, int n)
{
  Device * dev = (Device *) ptr;
  dev->init_old_sivecs_host(k, n);
}

/* ---------------------------------------------------------------------- */
void libgpu_push_sivecs_to_host(void * ptr, 
                                  py::array_t<double> vec, int loc, int size)
{
  Device * dev = (Device *) ptr;
  dev->push_sivecs_to_host(vec, loc, size);
}
/* ---------------------------------------------------------------------- */
void libgpu_push_instruction_list(void * ptr, 
                                  py::array_t<int> instruction_list, int len)
{
  Device * dev = (Device *) ptr;
  dev->push_instruction_list(instruction_list, len);
}

/* ---------------------------------------------------------------------- */
void libgpu_compute_sivecs(void * ptr, 
                             int m, int n, int k)
{
  Device * dev = (Device *) ptr;
  dev->compute_sivecs(m, n, k);
}
/* ---------------------------------------------------------------------- */
void libgpu_compute_sivecs_full(void * ptr, 
                             int m, int k, int counts)
{
  Device * dev = (Device *) ptr;
  dev->compute_sivecs_full(m, k, counts);
}
/* ---------------------------------------------------------------------- */
void libgpu_print_sivecs(void * ptr, int start, int size)
{
  Device * dev = (Device *) ptr;
  dev->print_sivecs(start, size);
}
/* ---------------------------------------------------------------------- */
void libgpu_pull_sivecs_from_pinned(void * ptr, 
                     py::array_t<double> vec, int n_loc, int m, int n)
{
  Device * dev = (Device *) ptr;
  dev->pull_sivecs_from_pinned(vec, n_loc, m, n);
}
/* ---------------------------------------------------------------------- */
void libgpu_add_ox1_pinned(void * ptr, 
                     py::array_t<double> ox1, int size)
{
  Device * dev = (Device *) ptr;
  dev->add_ox1_pinned(ox1, size);
}
/* ---------------------------------------------------------------------- */
void libgpu_finalize_ox1_pinned(void * ptr, 
                     py::array_t<double> ox1, int size)
{
  Device * dev = (Device *) ptr;
  dev->finalize_ox1_pinned(ox1, size);
}

/* ---------------------------------------------------------------------- */
