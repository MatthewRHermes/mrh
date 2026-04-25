
/*
This file is generalization of fci_contract_nosym.c to complex version. Instead of
calling the FCIContract_2es1 function four times, In this file I have written the FCIcontract_2es1_zgemm 
with complex arrays.

*/

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <complex.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "fci.h"
#define CSUMTHR         1e-28
// I have increase this to 240
#define STRB_BLKSIZE    240

double FCI_t1ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb);

static void zset0(double complex *x, size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                x[i] = 0.0 + 0.0 * I;
        }
}


static void pack_z_from_ri(double complex *z,
                           const double *r,
                           const double *im,
                           size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                z[i] = r[i] + im[i] * I;
        }
}


static void unpack_z_to_ri(const double complex *z,
                           double *r,
                           double *im,
                           size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                r[i]  = creal(z[i]);
                im[i] = cimag(z[i]);
        }
}


static void spread_a_t1_z(double complex *ci1, double complex *t1,
                          int bcount, int stra_id, int strb_id,
                          int norb, int nstrb, int nlinka,
                          _LinkT *clink_indexa)
{
        ci1 += strb_id;
        const int nnorb = norb * norb;
        int j, k, i, a, str1, sign;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double complex *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);

                cp0 = t1 + a*norb+i;
                cp1 = ci1 + str1*(size_t)nstrb;

                if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] += cp0[k*nnorb];
                        }
                } else {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] -= cp0[k*nnorb];
                        }
                }
        }
}


static void spread_b_t1_z(double complex *ci1, double complex *t1,
                          int bcount, int stra_id, int strb_id,
                          int norb, int nstrb, int nlinkb,
                          _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        int j, i, a, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double complex *pci = ci1 + stra_id * (size_t)nstrb;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);

                        pci[str1] += sign * t1[a*norb+i];
                }
                t1 += nnorb;
                tab += nlinkb;
        }
}


static void axpy2d_z(double complex *out,
                     double complex *in,
                     int count, int no, int ni)
{
        int i, j;
        for (i = 0; i < count; i++) {
                for (j = 0; j < ni; j++) {
                        out[i*(size_t)no+j] += in[i*(size_t)ni+j];
                }
        }
}

static void ctr_rhf2e_kern_zgemm(double complex *eri,
                                  double *ci0R, double *ci0I,
                                  double complex *ci1,
                                  double complex *ci1buf,
                                  double *t1R, double *t1I,
                                  double complex *t1Z,
                                  double complex *vt1Z,
                                  int bcount_for_spread_a,
                                  int ncol_ci1buf,
                                  int bcount,
                                  int stra_id,
                                  int strb_id,
                                  int norb,
                                  int na,
                                  int nb,
                                  int nlinka,
                                  int nlinkb,
                                  _LinkT *clink_indexa,
                                  _LinkT *clink_indexb,
                                  long long *n_zgemm_call,
                                  long long *n_skip)
{
        const char TRANS_N = 'N';
        const double complex Z0 = 0.0 + 0.0 * I;
        const double complex Z1 = 1.0 + 0.0 * I;
        const int nnorb = norb * norb;

        double csumR, csumI;
        
        NPdset0(t1R, ((size_t)nnorb) * bcount);
        NPdset0(t1I, ((size_t)nnorb) * bcount);

        csumR = FCI_t1ci_sf(ci0R, t1R, bcount, stra_id, strb_id,
                            norb, na, nb, nlinka, nlinkb,
                            clink_indexa, clink_indexb);

        csumI = FCI_t1ci_sf(ci0I, t1I, bcount, stra_id, strb_id,
                            norb, na, nb, nlinka, nlinkb,
                            clink_indexa, clink_indexb);
        if (csumR > CSUMTHR || csumI > CSUMTHR) {
        // if (csumR + csumI > CSUMTHR) {
                pack_z_from_ri(t1Z, t1R, t1I, (size_t)nnorb * bcount);

                zgemm_(&TRANS_N, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &Z1, eri, &nnorb,
                       t1Z, &nnorb,
                       &Z0, vt1Z, &nnorb);

                spread_b_t1_z(ci1, vt1Z, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);

                spread_a_t1_z(ci1buf, vt1Z, bcount_for_spread_a,
                              stra_id, 0,
                              norb, ncol_ci1buf, nlinka,
                              clink_indexa);

                *n_zgemm_call += 1;
        } else {
                *n_skip += 1;
        }
}

void FCIcontract_2es1_zgemm(double complex *eri,
                            double complex *ci0,
                            double complex *ci1,
                            int norb, int na, int nb,
                            int nlinka, int nlinkb,
                            int *link_indexa,
                            int *link_indexb)
{
        const int nnorb = norb * norb;
        const size_t ndet = ((size_t)na) * nb;

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);

        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);

        /*
         * FCI_t1ci_sf is a real PySCF helper.
         * So unpack complex CI into contiguous real/imag arrays once.
         */
        double *ci0R = malloc(sizeof(double) * ndet);
        double *ci0I = malloc(sizeof(double) * ndet);

        unpack_z_to_ri(ci0, ci0R, ci0I, ndet);

        zset0(ci1, ndet);

        long long n_zgemm_call_total = 0;
        long long n_skip_total = 0;

#pragma omp parallel default(none) \
        shared(eri, ci0R, ci0I, ci1, norb, na, nb, nnorb, nlinka, nlinkb, \
               clinka, clinkb, n_zgemm_call_total, n_skip_total)
{
        int strk, ib, blen;

        long long n_zgemm_call_private = 0;
        long long n_skip_private = 0;

        double *t1R = malloc(sizeof(double) * STRB_BLKSIZE * nnorb);
        double *t1I = malloc(sizeof(double) * STRB_BLKSIZE * nnorb);

        double complex *t1Z =
                malloc(sizeof(double complex) * STRB_BLKSIZE * nnorb);

        double complex *vt1Z =
                malloc(sizeof(double complex) * STRB_BLKSIZE * nnorb);

        double complex *ci1buf =
                malloc(sizeof(double complex) * (((size_t)na) * STRB_BLKSIZE + 2));

        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);

                zset0(ci1buf, ((size_t)na) * blen);

#pragma omp for schedule(static)
                for (strk = 0; strk < na; strk++) {
                        ctr_rhf2e_kern_zgemm(eri,
                                             ci0R, ci0I,
                                             ci1,
                                             ci1buf,
                                             t1R, t1I,
                                             t1Z, vt1Z,
                                             blen, blen, blen,
                                             strk, ib,
                                             norb, na, nb,
                                             nlinka, nlinkb,
                                             clinka, clinkb,
                                             &n_zgemm_call_private,
                                             &n_skip_private);
                }

#pragma omp critical
                {
                        axpy2d_z(ci1+ib, ci1buf, na, nb, blen);
                }

#pragma omp barrier
        }

        free(ci1buf);
        free(vt1Z);
        free(t1Z);
        free(t1I);
        free(t1R);

#pragma omp atomic
        n_zgemm_call_total += n_zgemm_call_private;

#pragma omp atomic
        n_skip_total += n_skip_private;
}

        free(ci0R);
        free(ci0I);
        free(clinka);
        free(clinkb);

}