/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf)
{
  // not used by the host backend
}

/* ---------------------------------------------------------------------- */
/* Host translations of the CUDA kernels in device_cuda.cpp
/* ---------------------------------------------------------------------- */

void Device::getjk_rho(double * rho, double * dmtril, double * eri, int nset, int naux, int nao_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nset; ++i) {
    for(int j=0; j<naux; ++j) {
      double val = 0.0;
      for(int k=0; k<nao_pair; ++k)
        val += dmtril[i*nao_pair + k] * eri[j*nao_pair + k];
      rho[i*naux + j] = val;
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::getjk_vj(double * vj, double * rho, double * eri, int nset, int nao_pair, int naux, int init)
{
  const int gs_nao_pair = (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE;
  const int chunk_size = (gs_nao_pair <= _CUDA_MAX_GRID_DIM_YZ) ? gs_nao_pair : _CUDA_MAX_GRID_DIM_YZ;
  const int num_chunks = (gs_nao_pair <= _CUDA_MAX_GRID_DIM_YZ) ? 1 : (gs_nao_pair / _CUDA_MAX_GRID_DIM_YZ + 1);
  const int z_block = chunk_size * _DOT_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
  for(int i=0; i<nset; ++i) {
    for(int j=0; j<num_chunks; ++j) {
      for(int k=0; k<z_block; ++k) {
        int indxK = j*chunk_size + k;
        if(indxK >= nao_pair) continue;
        double val = 0.0;
        for(int l=0; l<naux; ++l) val += rho[i*naux + l] * eri[l*nao_pair + indxK];
        if(init) vj[i*nao_pair + indxK] = val;
        else vj[i*nao_pair + indxK] += val;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::getjk_unpack_buf2(double * buf2, double * eri, int * map, int naux, int nao, int nao_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i) {
    for(int j=0; j<nao; ++j) {
      double * buf = &(buf2[i*nao*nao]);
      double * tril = &(eri[i*nao_pair]);
      const int indx = j*nao;
      for(int k=0; k<nao; ++k) buf[indx+k] = tril[map[indx+k]];
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::pack_eri(double * eri1, double * buf2, int * map, int naux, int nao, int nao_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i) {
    for(int j=0; j<nao; ++j) {
      double * buf = &(buf2[i*nao*nao]);
      double * tril = &(eri1[i*nao_pair]);
      const int indx = j*nao;
      for(int k=0; k<nao; ++k) tril[map[indx+k]] = buf[indx+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::transpose(double * out, double * in, int nrow, int ncol)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nrow; ++i)
    for(int j=0; j<ncol; ++j)
      out[j*nrow + i] = in[i*ncol + j];
}

/* ---------------------------------------------------------------------- */

void Device::get_bufpa(const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<nmo; ++j)
      for(int k=0; k<ncas; ++k) {
        int inputIndex = (i*nmo + j)*nmo + k+ncore;
        int outputIndex = (i*nmo + j)*ncas + k;
        bufpa[outputIndex] = bufpp[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::get_bufaa(const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k) {
        int inputIndex = (i*nmo + (j+ncore))*nmo + k+ncore;
        int outputIndex = (i*ncas + j)*ncas + k;
        bufaa[outputIndex] = bufpp[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_120(double * in, double * out, int naux, int nao, int ncas, int order)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<nao; ++k) {
        int inputIndex = i*nao*ncas + j*nao + k;
        int outputIndex = j*nao*naux + k*naux + i;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::get_bufd(const double* bufpp, double* bufd, int naux, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<nmo; ++j)
      bufd[i*nmo + j] = bufpp[(i*nmo + j)*nmo + j];
}

/* ---------------------------------------------------------------------- */

void Device::transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<nao; ++k) {
        int inputIndex = i*nao*ncas + j*nao + k;
        int outputIndex = k*ncas*naux + j*naux + i;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::extract_submatrix(const double* big_mat, double* small_mat, int ncas, int ncore, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<ncas; ++i)
    for(int j=0; j<ncas; ++j)
      small_mat[i*ncas + j] = big_mat[(i+ncore)*nmo + (j+ncore)];
}

/* ---------------------------------------------------------------------- */

void Device::unpack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nmo*ncas; ++i)
    for(int j=0; j<ncas*ncas; ++j) {
      double * in_buf = &(in[i*ncas_pair]);
      double * out_buf = &(out[i*ncas*ncas]);
      out_buf[j] = in_buf[map[j]];
    }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_2310(double * in, double * out, int nmo, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<nmo; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k)
        for(int l=0; l<ncas; ++l) {
          int inputIndex = ((i*ncas+j)*ncas+k)*ncas+l;
          int outputIndex = k*ncas*ncas*nmo + l*ncas*nmo + j*nmo + i;
          out[outputIndex] = in[inputIndex];
        }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_3210(double* in, double* out, int nmo, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ncas; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k)
        for(int l=0; l<nmo; ++l) {
          int inputIndex = ((i*ncas+j)*ncas+k)*nmo+l;
          int outputIndex = l*ncas*ncas*ncas + k*ncas*ncas + j*ncas + i;
          out[outputIndex] = in[inputIndex];
        }
}

/* ---------------------------------------------------------------------- */

void Device::pack_h2eff_2d(double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<nmo; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas_pair; ++k) {
        double * out_buf = &(out[(i*ncas + j)*ncas_pair]);
        double * in_buf = &(in[(i*ncas + j)*ncas*ncas]);
        out_buf[k] = in_buf[map[k]];
      }
}

/* ---------------------------------------------------------------------- */

void Device::get_mo_cas(const double* big_mat, double* small_mat, int ncas, int ncore, int nao, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<ncas; ++i)
    for(int j=0; j<nao; ++j)
      small_mat[i*nao + j] = big_mat[j*nmo + i+ncore];
}

/* ---------------------------------------------------------------------- */

void Device::pack_d_vuwM(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<nmo*ncas; ++i)
    for(int j=0; j<ncas*ncas; ++j)
      out[i*ncas_pair + map[j]] = in[j*ncas*nmo + i];
}

/* ---------------------------------------------------------------------- */

void Device::pack_d_vuwM_add(const double * in, double * out, int * map, int nmo, int ncas, int ncas_pair)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<nmo*ncas; ++i)
    for(int j=0; j<ncas*ncas; ++j)
      out[i*ncas_pair + map[j]] += in[j*ncas*nmo + i];
}

/* ---------------------------------------------------------------------- */

void Device::vecadd(const double * in, double * out, int N)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<N; ++i) out[i] += in[i];
}

/* ---------------------------------------------------------------------- */

void Device::vecadd_batch(const double * in, double * out, int N, int num_batches)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<N; ++i) {
    double val = 0.0;
    for(int j=0; j<num_batches; ++j) val += in[j*N + i];
    out[i] += val;
  }
}

/* ---------------------------------------------------------------------- */

void Device::memset_zero_batch_stride(double * inout, int stride, int offset, int N, int num_batches)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<N; ++i)
    for(int j=0; j<num_batches; ++j) inout[j*stride + offset + i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void Device::get_rho_to_Pi(double * rho, double * Pi, int ngrid)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<ngrid; ++i) Pi[i] += rho[i]*rho[i];
}

/* ---------------------------------------------------------------------- */

void Device::make_gridkern(double * mo_grid, double * gridkern, int ngrid, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ngrid; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k) {
        double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
        double * tmp_mo_grid = &(mo_grid[i*ncas]);
        tmp_gridkern[j*ncas+k] = tmp_mo_grid[j]*tmp_mo_grid[k];
      }
}

/* ---------------------------------------------------------------------- */

void Device::make_buf_pdft(double * gridkern, double * buf, double * cascm2, int ngrid, int ncas)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<ngrid; ++i)
    for(int j=0; j<ncas*ncas; ++j) {
      double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
      double * tmp_cascm2 = &(cascm2[j*ncas*ncas]);
      double * tmp_out = &(buf[i*ncas*ncas+j]);
      for(int k=0; k<ncas*ncas; ++k) tmp_out[0] += tmp_gridkern[k]*tmp_cascm2[k];
    }
}

/* ---------------------------------------------------------------------- */

void Device::make_Pi_final(double * gridkern, double * buf, double * Pi, int ngrid, int ncas)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<ngrid; ++i) {
    double * tmp_gridkern = &(gridkern[i*ncas*ncas]);
    double * tmp_buf = &(buf[i*ncas*ncas]);
    double * tmp_Pi = &(Pi[i]);
    for(int j=0; j<ncas*ncas; ++j) tmp_Pi[0] += tmp_gridkern[j]*tmp_buf[j];
  }
}

/* ---------------------------------------------------------------------- */

void Device::set_to_zero(double * array, int size)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<size; ++i) array[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCItrans_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int str0=0; str0<na; ++str0)
    for(int j=0; j<nlinka; ++j) {
      int * tab = &(link_index[4*nlinka*str0 + 4*j]);
      int a = tab[0];
      int i = tab[1];
      int str1 = tab[2];
      int sign = tab[3];
      if(sign == 0) continue;
      double * pket = &(ciket[str0*nb]);
      double * pbra = &(cibra[str1*nb]);
      for(int k=0; k<nb; ++k) {
#pragma omp atomic
        rdm[a*norb+i] += sign*pbra[k]*pket[k];
      }
    }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCItrans_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int str0=0; str0<na; ++str0)
    for(int k=0; k<nb; ++k)
      for(int j=0; j<nlinkb; ++j) {
        double * pbra = &(cibra[str0*nb]);
        double tmp = ciket[str0*nb + k];
        int * tab = &(link_index[4*nlinkb*k + 4*j]);
        int a = tab[0];
        int i = tab[1];
        int str1 = tab[2];
        int sign = tab[3];
#pragma omp atomic
        rdm[a*norb + i] += sign*pbra[str1]*tmp;
      }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCItrans_rdm1a_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinka,
                                        int ia_bra, int ja_bra, int ib_bra, int jb_bra,
                                        int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sign_dummy,
                                        int * link_index)
{
  int na_bra = ja_bra - ia_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_bra = jb_bra - ib_bra;
  int nb_ket = jb_ket - ib_ket;
  int ib_max = (ib_bra > ib_ket) ? ib_bra : ib_ket;
  int jb_min = (jb_bra < jb_ket) ? jb_bra : jb_ket;
  int b_len = jb_min - ib_max;
  int b_bra_offset = ib_max - ib_bra;
  int b_ket_offset = ib_max - ib_ket;
  if(b_len <= 0) return;

#pragma omp parallel for collapse(2) schedule(static)
  for(int str0=0; str0<na_ket; ++str0)
    for(int j=0; j<nlinka; ++j) {
      int * tab = &(link_index[4*nlinka*(str0+ia_ket) + 4*j]);
      int sign = tab[3];
      if(sign == 0) continue;
      sign = sign * sign_dummy;
      int str1 = tab[2];
      if((str1 >= ia_bra) && (str1 < ja_bra)) {
        int a = tab[0];
        int i = tab[1];
        double * pket = &(ciket[str0*nb_ket]);
        double * pbra = &(cibra[(str1-ia_bra)*nb_bra]);
        for(int k=0; k<b_len; ++k) {
#pragma omp atomic
          rdm[a*norb+i] += sign*pbra[k+b_bra_offset]*pket[k+b_ket_offset];
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCItrans_rdm1b_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinkb,
                                        int ia_bra, int ja_bra, int ib_bra, int jb_bra,
                                        int ia_ket, int ja_ket, int ib_ket, int jb_ket, int sign_dummy,
                                        int * link_index)
{
  int na_bra = ja_bra - ia_bra;
  int na_ket = ja_ket - ia_ket;
  int nb_bra = jb_bra - ib_bra;
  int nb_ket = jb_ket - ib_ket;
  int ia_max = (ia_bra > ia_ket) ? ia_bra : ia_ket;
  int ja_min = (ja_bra < ja_ket) ? ja_bra : ja_ket;
  int a_len = ja_min - ia_max;
  if(a_len <= 0) return;

#pragma omp parallel for collapse(3) schedule(static)
  for(int str0=0; str0<a_len; ++str0)
    for(int k=0; k<nb_ket; ++k)
      for(int j=0; j<nlinkb; ++j) {
        double * pbra = &(cibra[(str0+ia_max)*nb_bra]);
        double tmp = ciket[(str0+ia_max)*nb_ket + k];
        int * tab = &(link_index[4*nlinkb*(k+ib_ket) + 4*j]);
        int str1 = tab[2];
        if((str1 >= ib_bra) && (str1 < jb_bra)) {
          int sign = tab[3];
          if(sign == 0) continue;
          sign = sign * sign_dummy;
          int a = tab[0];
          int i = tab[1];
#pragma omp atomic
          rdm[a*norb + i] += sign*pbra[str1-ib_bra]*tmp;
        }
      }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCImake_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int str0=0; str0<na; ++str0)
    for(int j=0; j<nlinka; ++j) {
      double * pci0 = &(ciket[str0*nb]);
      int * tab = &(link_index[4*nlinka*str0 + 4*j]);
      int a = tab[0];
      int i = tab[1];
      int str1 = tab[2];
      int sign = tab[3];
      double * pci1 = &(ciket[str1*nb]);
      if(a >= i && sign != 0) {
        for(int k=0; k<nb; ++k) {
#pragma omp atomic
          rdm[a*norb+i] += sign*pci0[k]*pci1[k];
        }
      }
    }

  for(int i=0; i<norb; ++i)
    for(int j=0; j<i; ++j)
      rdm[j*norb+i] = rdm[i*norb+j];
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCImake_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int str0=0; str0<na; ++str0)
    for(int k=0; k<nb; ++k)
      for(int j=0; j<nlinkb; ++j) {
        double * pci0 = &(ciket[str0*nb]);
        int * tab = &(link_index[4*nlinkb*k + 4*j]);
        int a = tab[0];
        int i = tab[1];
        int sign = tab[3];
        if(a >= i && sign != 0) {
          int str1 = tab[2];
#pragma omp atomic
          rdm[a*norb+i] += sign*pci0[str1]*pci0[k];
        }
      }

  for(int i=0; i<norb; ++i)
    for(int j=0; j<i; ++j)
      rdm[j*norb+i] = rdm[i*norb+j];
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCIrdm2_a_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinka, int * link_index)
{
  int norb2 = norb*norb;
#pragma omp parallel for collapse(2) schedule(static)
  for(int batch_id=0; batch_id<batches; ++batch_id)
    for(int k=0; k<nb; ++k) {
      int * tab_line = &(link_index[4*nlinka*(stra_id+batch_id)]);
      double * tmp_buf = &(buf[batch_id*norb2*nb + k*norb2]);
      for(int j=0; j<nlinka; ++j) {
        int * tab = &(tab_line[4*j]);
        int sign = tab[3];
        if(sign != 0) {
          int a = tab[0];
          int i = tab[1];
          int str1 = tab[2];
          tmp_buf[i*norb + a] += sign*ci[str1*nb + k];
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCIrdm2_b_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinkb, int * link_index)
{
  int norb2 = norb*norb;
#pragma omp parallel for collapse(2) schedule(static)
  for(int batch_id=0; batch_id<batches; ++batch_id)
    for(int str0=0; str0<nb; ++str0) {
      double * tmp_buf = &(buf[batch_id*norb2*nb + str0*norb2]);
      int * tab_line = &(link_index[4*str0*nlinkb]);
      double * tmp_ci = &(ci[(stra_id+batch_id)*nb]);
      for(int j=0; j<nlinkb; ++j) {
        int * tab = &(tab_line[4*j]);
        int sign = tab[3];
        if(sign != 0) {
          int a = tab[0];
          int i = tab[1];
          int str1 = tab[2];
          tmp_buf[i*norb + a] += sign*tmp_ci[str1];
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCIrdm3h_a_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  int norb2 = norb*norb;
#pragma omp parallel for schedule(static)
  for(int k=0; k<jb-ib; ++k) {
    double * tmp_buf = &(buf[(k+ib)*norb2]);
    int * tab_line = &(link_index[4*nlinka*stra_id]);
    for(int j=0; j<nlinka; ++j) {
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if(sign != 0) {
        int str1 = tab[2];
        if((str1 >= ia) && (str1 < ja)) {
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(str1-ia)*nb + k];
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCIrdm3h_b_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  int norb2 = norb*norb;
#pragma omp parallel for schedule(static)
  for(int str0=0; str0<nb; ++str0) {
    double * tmp_buf = &(buf[str0*norb2]);
    int * tab_line = &(link_index[4*str0*nlinkb]);
    for(int j=0; j<nlinkb; ++j) {
      int * tab = &(tab_line[4*j]);
      int sign = tab[3];
      if(sign != 0) {
        int str1 = tab[2];
        if((str1 >= ib) && (str1 < jb)) {
          int a = tab[0];
          int i = tab[1];
          tmp_buf[i*norb + a] += sign*ci[(stra_id-ia)*nb_bra + str1-ib];
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCIrdm3h_a_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_ci, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
{
  int norb2 = norb*norb;
#pragma omp parallel for collapse(2) schedule(static)
  for(int batch_id=0; batch_id<batches; ++batch_id)
    for(int k=0; k<jb-ib; ++k) {
      double * tmp_buf = &(buf[batch_id*norb2*nb + (k+ib)*norb2]);
      int * tab_line = &(link_index[4*nlinka*(stra_id+batch_id)]);
      for(int j=0; j<nlinka; ++j) {
        int * tab = &(tab_line[4*j]);
        int sign = tab[3];
        if(sign != 0) {
          int str1 = tab[2];
          if((str1 >= ia) && (str1 < ja)) {
            int a = tab[0];
            int i = tab[1];
            tmp_buf[i*norb + a] += sign*ci[(str1-ia)*nb_ci + k];
          }
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

void Device::compute_FCIrdm3h_b_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
{
  int norb2 = norb*norb;
#pragma omp parallel for collapse(2) schedule(static)
  for(int batch_id=0; batch_id<batches; ++batch_id)
    for(int str0=0; str0<nb; ++str0) {
      double * tmp_buf = &(buf[batch_id*norb2*nb + str0*norb2]);
      int * tab_line = &(link_index[4*str0*nlinkb]);
      for(int j=0; j<nlinkb; ++j) {
        int * tab = &(tab_line[4*j]);
        int sign = tab[3];
        if(sign != 0) {
          int str1 = tab[2];
          if((str1 >= ib) && (str1 < jb)) {
            int a = tab[0];
            int i = tab[1];
            tmp_buf[i*norb + a] += sign*ci[(stra_id+batch_id-ia)*nb_bra + str1-ib];
          }
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_jikl(double * tdm, double * buf, int norb)
{
  int norb2 = norb*norb;
#pragma omp parallel for schedule(static)
  for(int k=0; k<norb2; ++k) {
    for(int i=0; i<norb; ++i)
      for(int j=0; j<norb; ++j) {
        const double * tmp_in = &(tdm[(i*norb+j)*norb2]);
        double * tmp_out = &(buf[(j*norb+i)*norb2]);
        tmp_out[k] = tmp_in[k];
      }
  }

#pragma omp parallel for schedule(static)
  for(int i=0; i<norb2*norb2; ++i) tdm[i] = buf[i];
}

/* ---------------------------------------------------------------------- */

void Device::reduce_buf3_to_rdm(const double * buf3, double * dm2, int size_tdm2, int num_gemm_batches)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<size_tdm2; ++i) {
    double val = 0.0;
    for(int j=0; j<num_gemm_batches; ++j) val += buf3[j*size_tdm2 + i];
    dm2[i] += val;
  }
}

/* ---------------------------------------------------------------------- */

void Device::reorder(double * dm1, double * dm2, double * buf, int norb)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<norb; ++i)
    for(int j=0; j<norb; ++j)
      for(int k=0; k<norb; ++k)
        dm2[((i*norb+j)*norb+j)*norb + k] -= dm1[i*norb + k];
}

/* ---------------------------------------------------------------------- */

void Device::filter_sfudm(const double * dm2, double * dm1, int norb)
{
  int norb_m1 = norb-1;
  int norb1 = norb_m1 + 1;
  int norb12 = (norb_m1+1)*(norb_m1+1);
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<norb_m1; ++i)
    for(int j=0; j<norb_m1; ++j)
      dm1[i*norb_m1+j] = dm2[i*norb12 + j*norb1 + norb_m1];
}

/* ---------------------------------------------------------------------- */

void Device::filter_tdmpp(const double * dm2, double * dm1, int norb, int spin)
{
  int ndum = (spin!=1) ? 2:1;
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<norb-ndum; ++i)
    for(int j=0; j<norb-ndum; ++j)
      dm1[i*(norb-ndum)+j] = dm2[i*norb*norb*norb + (norb-1)*norb*norb + j*norb + norb-ndum];
}

/* ---------------------------------------------------------------------- */

void Device::filter_tdm1h(const double * in, double * out, int norb)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<norb; ++i) out[i] = in[i*(norb+1)+norb];
}

/* ---------------------------------------------------------------------- */

void Device::filter_tdm3h(double * in, double * out, int norb)
{
  int norb1 = norb+1;
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<norb; ++i)
    for(int j=0; j<norb; ++j)
      for(int k=0; k<norb; ++k)
        out[(i*norb+j)*norb+k] = in[((i*norb1+norb)*norb1+j)*norb1+k];
}

/* ---------------------------------------------------------------------- */

void Device::veccopy(const double * src, double *dest, int size)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<size; ++i) dest[i] = src[i];
}

/* ---------------------------------------------------------------------- */

void Device::transpose_021(double * in, double * out, int ax1, int ax2, int ax3)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ax1; ++i)
    for(int j=0; j<ax2; ++j)
      for(int k=0; k<ax3; ++k) {
        int inputIndex = (i*ax3+k)*ax2+j;
        int outputIndex = (i*ax2+j)*ax3+k;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_102(double * in, double * out, int ax1, int ax2, int ax3)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<ax1; ++i)
    for(int j=0; j<ax2; ++j)
      for(int k=0; k<ax3; ++k) {
        int inputIndex = (j*ax1+i)*ax3+k;
        int outputIndex = (i*ax2+j)*ax3+k;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_2130(const double * in, double * out, int ax1, int ax2, int ax3, int ax4)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int idx1=0; idx1<ax1; ++idx1)
    for(int idx2=0; idx2<ax2; ++idx2)
      for(int idx3=0; idx3<ax3; ++idx3)
        for(int idx4=0; idx4<ax4; ++idx4) {
          int outputIndex = ((idx3*ax2 + idx2)*ax4 + idx4)*ax1 + idx1;
          int inputIndex = ((idx1*ax2 + idx2)*ax3 + idx3)*ax4 + idx4;
          out[outputIndex] = in[inputIndex];
        }
}

/* ---------------------------------------------------------------------- */

#endif
