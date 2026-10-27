/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void DeviceFci::compute_FCItrans_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
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

void DeviceFci::compute_FCItrans_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
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

void DeviceFci::compute_FCItrans_rdm1a_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinka,
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

void DeviceFci::compute_FCItrans_rdm1b_v2(double * cibra, double * ciket, double * rdm, int norb, int nlinkb,
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

void DeviceFci::compute_FCImake_rdm1a(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinka, int * link_index)
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

void DeviceFci::compute_FCImake_rdm1b(double * cibra, double * ciket, double * rdm, int norb, int na, int nb, int nlinkb, int * link_index)
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

void DeviceFci::compute_FCIrdm2_a_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinka, int * link_index)
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

void DeviceFci::compute_FCIrdm2_b_t1ci_v2(double * ci, double * buf, int stra_id, int batches, int nb, int norb, int nlinkb, int * link_index)
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

void DeviceFci::compute_FCIrdm3h_a_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
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

void DeviceFci::compute_FCIrdm3h_b_t1ci_v2(double * ci, double * buf, int stra_id, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
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

void DeviceFci::compute_FCIrdm3h_a_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_ci, int norb, int nlinka, int ia, int ja, int ib, int jb, int * link_index)
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

void DeviceFci::compute_FCIrdm3h_b_t1ci_v3(double * ci, double * buf, int stra_id, int batches, int nb, int nb_bra, int norb, int nlinkb, int ia, int ja, int ib, int jb, int * link_index)
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

void DeviceFci::transpose_jikl(double * tdm, double * buf, int norb)
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

void DeviceFci::reduce_buf3_to_rdm(const double * buf3, double * dm2, int size_tdm2, int num_gemm_batches)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<size_tdm2; ++i) {
    double val = 0.0;
    for(int j=0; j<num_gemm_batches; ++j) val += buf3[j*size_tdm2 + i];
    dm2[i] += val;
  }
}

/* ---------------------------------------------------------------------- */

void DeviceFci::reorder(double * dm1, double * dm2, double * buf, int norb)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<norb; ++i)
    for(int j=0; j<norb; ++j)
      for(int k=0; k<norb; ++k)
        dm2[((i*norb+j)*norb+j)*norb + k] -= dm1[i*norb + k];
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_sfudm(const double * dm2, double * dm1, int norb)
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

void DeviceFci::filter_tdmpp(const double * dm2, double * dm1, int norb, int spin)
{
  int ndum = (spin!=1) ? 2:1;
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<norb-ndum; ++i)
    for(int j=0; j<norb-ndum; ++j)
      dm1[i*(norb-ndum)+j] = dm2[i*norb*norb*norb + (norb-1)*norb*norb + j*norb + norb-ndum];
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_tdm1h(const double * in, double * out, int norb)
{
#pragma omp parallel for schedule(static)
  for(int i=0; i<norb; ++i) out[i] = in[i*(norb+1)+norb];
}

/* ---------------------------------------------------------------------- */

void DeviceFci::filter_tdm3h(double * in, double * out, int norb)
{
  int norb1 = norb+1;
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<norb; ++i)
    for(int j=0; j<norb; ++j)
      for(int k=0; k<norb; ++k)
        out[(i*norb+j)*norb+k] = in[((i*norb1+norb)*norb1+j)*norb1+k];
}


#endif
