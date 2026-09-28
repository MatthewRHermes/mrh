/*
* Author: Bhavnesh Jangid
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <complex.h>
#include "config.h"
#include "vhf/fblas.h"
#include "fci.h"
#include "np_helper/np_helper.h"

#define CSUMTHR         1e-28
#define BUFBASE         96
#define SQRT2           1.4142135623730950488

#define BRAKETSYM       1
#define PARTICLESYM     2

#define KBLOCK_KA       0
#define KBLOCK_KB       1
#define KBLOCK_NA       2
#define KBLOCK_NB       3
#define KBLOCK_OFFSET   4
#define KBLOCK_SIZE     5

void FCIkci_sector_to_full_cplx(double complex *ci_full,
                                double complex *ci_sector,
                                int nblocks, int *blocks,
                                int *stra_ids, int *stra_offsets,
                                int *strb_ids, int *strb_offsets,
                                int na_full, int nb_full)
{
        int iblk, ia, ib;
        int ka, kb, na_blk, nb_blk, offset;
        int astr, bstr;

        NPzset0(ci_full, na_full * nb_full);

        for (iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                ka = blk[KBLOCK_KA];
                kb = blk[KBLOCK_KB];
                na_blk = blk[KBLOCK_NA];
                nb_blk = blk[KBLOCK_NB];
                offset = blk[KBLOCK_OFFSET];

                for (ia = 0; ia < na_blk; ia++) {
                        astr = stra_ids[stra_offsets[ka] + ia];
                        for (ib = 0; ib < nb_blk; ib++) {
                                bstr = strb_ids[strb_offsets[kb] + ib];
                                ci_full[(size_t)astr * nb_full + bstr] =
                                        ci_sector[offset + ia * nb_blk + ib];
                        }
                }
        }
}

void FCIkci_full_to_sector_cplx(double complex *ci_sector,
                                double complex *ci_full,
                                int nblocks, int *blocks,
                                int *stra_ids, int *stra_offsets,
                                int *strb_ids, int *strb_offsets,
                                int nb_full)
{
        int iblk, ia, ib;
        int ka, kb, na_blk, nb_blk, offset, size;
        int astr, bstr;

        for (iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                size = blk[KBLOCK_SIZE];
                offset = blk[KBLOCK_OFFSET];
                NPzset0(ci_sector + offset, size);

                ka = blk[KBLOCK_KA];
                kb = blk[KBLOCK_KB];
                na_blk = blk[KBLOCK_NA];
                nb_blk = blk[KBLOCK_NB];

                for (ia = 0; ia < na_blk; ia++) {
                        astr = stra_ids[stra_offsets[ka] + ia];
                        for (ib = 0; ib < nb_blk; ib++) {
                                bstr = strb_ids[strb_offsets[kb] + ib];
                                ci_sector[offset + ia * nb_blk + ib] =
                                        ci_full[(size_t)astr * nb_full + bstr];
                        }
                }
        }
}

extern void zherk_(const char*, const char*, const int*, const int*,
                   const double*, const double complex*, const int*,
                   const double*, double complex*, const int*);
          
typedef void (*dm12kernel_cplx_t)(double complex*, double complex*,
                                 double complex*, double complex*,
                                 int, int, int,
                                 int, int, int, int, int,
                                 _LinkT*, _LinkT*, int);

                                 
/*
 * make_rdm1 for complex FCI wavefunction
 */
void FCImake_rdm1a_cplx(double complex *rdm1,
                       double complex *cibra,
                       double complex *ciket,
                       int norb, int na, int nb, int nlinka, int nlinkb,
                       int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double complex *pci0, *pci1;
        double complex *ci0 = ciket;

        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinka * na);
        FCIcompress_link(clink, link_indexa, norb, na, nlinka);

        NPzset0(rdm1, norb*norb);
        
        for (str0 = 0; str0 < na; str0++) {
                tab = clink + str0 * nlinka;
                pci0 = ci0 + str0 * nb;
                for (j = 0; j < nlinka; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pci1 = ci0 + str1 * nb;
                        if (a >= i) {
                                if (sign == 0){
                                        break;
                                }
                                else if (sign > 0) {
                                        for (k = 0; k < nb; k++) {
                                                rdm1[a*norb+i] += conj(pci0[k]) * pci1[k];
                                        }
                                }
                                else {
                                        for (k = 0; k < nb; k++) {
                                                rdm1[a*norb+i] -= conj(pci0[k]) * pci1[k];
                                        }
                                }
                        }
                }
        }
    
    for (j = 0; j < norb; j++) {        
        for (k = 0; k < j; k++) {
            rdm1[k*norb+j] = conj(rdm1[j*norb+k]);
        }
    }
    
    free(clink);
}


void FCImake_rdm1b_cplx(double complex *rdm1,
                        double complex *cibra,
                        double complex *ciket,
                        int norb, int na, int nb, int nlinka, int nlinkb,
                
                        int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double complex *pci0, *pci1;
        double complex *ci0 = ciket;
        double complex tmp;

        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clink, link_indexb, norb, nb, nlinkb);

        NPzset0(rdm1, norb*norb);
        
        for (str0 = 0; str0 < na; str0++) {
                pci0 = ci0 + str0 * nb;
                for (k = 0; k < nb; k++) {
                        tab = clink + k * nlinkb;
                        tmp = pci0[k];
                        for (j = 0; j < nlinkb; j++) {
                                a    = EXTRACT_CRE (tab[j]);
                                i    = EXTRACT_DES (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                if (a >= i) {
                                        if (sign == 0){
                                                break;
                                        }
                                        else if (sign > 0){
                                                rdm1[a*norb+i] += pci0[str1] * conj(tmp);
                                        }
                                        else{
                                                rdm1[a*norb+i] -= pci0[str1] * conj(tmp);
                                        }
                                }
                        }
                }
        }
        
        for (j = 0; j < norb; j++) {
                for (k = 0; k < j; k++) {
                        rdm1[k*norb+j] = conj(rdm1[j*norb+k]);
        }
}

    free(clink);
}


/*
 * Complex transition 1-RDMs.  Unlike FCImake_rdm1[ab]_cplx, these routines
 * use both CI vectors, evaluate every orbital pair, and do not impose
 * Hermiticity. Note: in python wrapper, we transpose it to get the respective
 * rdm1[i, a] matrix (not conjugate transpose).  The matrix elements are defined as
 *
 *     rdm1[a,i] = <cibra|a^+_a a_i|ciket>.
 */
void FCItrans_rdm1a_cplx(double complex *rdm1,
                         double complex *cibra,
                         double complex *ciket,
                         int norb, int na, int nb, int nlinka, int nlinkb,
                         int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double complex *pket, *pbra;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * (size_t)nlinka * (size_t)na);
        FCIcompress_link(clink, link_indexa, norb, na, nlinka);

        NPzset0(rdm1, norb*norb);
        for (str0 = 0; str0 < na; str0++) {
                tab = clink + str0 * nlinka;
                pket = ciket + (size_t)str0 * (size_t)nb;
                for (j = 0; j < nlinka; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        if (sign == 0) {
                                break;
                        }
                        pbra = cibra + (size_t)str1 * (size_t)nb;
                        for (k = 0; k < nb; k++) {
                                rdm1[a*norb+i] += ((double)sign)
                                                * conj(pbra[k]) * pket[k];
                        }
                }
        }
        free(clink);
}


void FCItrans_rdm1b_cplx(double complex *rdm1,
                         double complex *cibra,
                         double complex *ciket,
                         int norb, int na, int nb, int nlinka, int nlinkb,
                         int *link_indexa, int *link_indexb)
{
        int i, a, j, k, stra, str1, sign;
        double complex *pket, *pbra;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * (size_t)nlinkb * (size_t)nb);
        FCIcompress_link(clink, link_indexb, norb, nb, nlinkb);

        NPzset0(rdm1, norb*norb);
        for (stra = 0; stra < na; stra++) {
                pbra = cibra + (size_t)stra * (size_t)nb;
                pket = ciket + (size_t)stra * (size_t)nb;
                for (k = 0; k < nb; k++) {
                        tab = clink + k * nlinkb;
                        for (j = 0; j < nlinkb; j++) {
                                a    = EXTRACT_CRE (tab[j]);
                                i    = EXTRACT_DES (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                if (sign == 0) {
                                        break;
                                }
                                rdm1[a*norb+i] += ((double)sign)
                                                * conj(pbra[str1]) * pket[k];
                        }
                }
        }
        free(clink);
}


static void _transpose_jikl_cplx(double complex *dm2, int norb)
{
        int nnorb = norb * norb;
        int i, j, k;
        double complex *p0, *p1;
        double complex *tmp = malloc(sizeof(double complex) * (size_t)nnorb * (size_t)nnorb);

        NPzcopy(tmp, dm2, nnorb*nnorb);
        
        for (i = 0; i < norb; i++){
                for (j = 0; j < norb; j++) {
                        p0 = tmp + (size_t)(j*norb+i) * (size_t)nnorb;
                        p1 = dm2 + (size_t)(i*norb+j) * (size_t)nnorb;
                        for (k = 0; k < nnorb; k++) {
                                p1[k] = p0[k];
                        }
                }
        }
        free(tmp);
}

/*
 * This is driver function, basically the threading and looping is controlled here
 * This also symmetrizes the final rdm2 according to the symmetry choice
 */
void FCIrdm12_drv_cplx(dm12kernel_cplx_t dm12kernel,
                       double complex *rdm1, double complex *rdm2,
                       double complex *bra,  double complex *ket,
                       int norb, int na, int nb, int nlinka, int nlinkb,
                       int *link_indexa, int *link_indexb, int symm)
{
        const int nnorb = norb * norb;
        int strk, i, j, k, l, ib, blen;
        double complex *pdm1, *pdm2;

        NPzset0(rdm1, nnorb);
        NPzset0(rdm2, nnorb*nnorb);

        _LinkT *clinka = malloc(sizeof(_LinkT) * (size_t)nlinka * (size_t)na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * (size_t)nlinkb * (size_t)nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);

#pragma omp parallel private(strk, i, j, k, l, ib, blen, pdm1, pdm2)
{
        pdm1 = calloc((size_t)nnorb + 2, sizeof(double complex));
        pdm2 = calloc((size_t)nnorb*(size_t)nnorb + 2, sizeof(double complex));

#pragma omp for schedule(dynamic, 40)
        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < nb; ib += BUFBASE) {
                        blen = MIN(BUFBASE, nb-ib);
                        (*dm12kernel)(pdm1, pdm2, bra, ket, blen, strk, ib,
                                norb, na, nb, nlinka, nlinkb,
                                clinka, clinkb, symm);
                        }
        }

#pragma omp critical
{
        for (i = 0; i < nnorb; i++) {
                rdm1[i] += pdm1[i];
        }
        for (i = 0; i < nnorb*nnorb; i++) {
                rdm2[i] += pdm2[i];
        }
}
        free(pdm1);
        free(pdm2);
}

        free(clinka);
        free(clinkb);

        switch (symm) {
        case BRAKETSYM:
                for (i = 0; i < norb; i++) {
                        for (j = 0; j < i; j++) {
                                rdm1[j*norb+i] = conj(rdm1[i*norb+j]);
                        }
                }

                for (i = 0; i < nnorb; i++) {
                        for (j = 0; j < i; j++) {
                                rdm2[j*nnorb+i] = conj(rdm2[i*nnorb+j]);
                        }
                }
                _transpose_jikl_cplx(rdm2, norb);
                break;
        
        // I didn't test this yet!
        case PARTICLESYM:
                for (i = 0; i < norb; i++) {
                        for (j = 0; j < i; j++) {
                                double complex *blk1 = rdm2 + (size_t)(i*nnorb+j) * (size_t)norb;
                                double complex *blk2 = rdm2 + (size_t)(j*nnorb+i) * (size_t)norb;
                                for (k = 0; k < norb; k++) {
                                        for (l = 0; l < norb; l++) {
                                                blk2[l*nnorb+k] = blk1[k*nnorb+l];
                                        }
                                }
                                for (k = 0; k < norb; k++) {
                                        blk2[i*nnorb+k] += rdm1[j*norb+k];
                                        blk2[k*nnorb+j] -= rdm1[i*norb+k];
                                }
                        } 
                }
                break;
        default:
                _transpose_jikl_cplx(rdm2, norb);
        }
}

/*
Helper function for 2pdm construction
*/
static void tril_particle_symm_cplx(double complex *rdm2, double complex *tbra,
                                    double complex *tket, int bcount, int norb, 
                                    double alpha, double beta)
{
        const char TRANS_N = 'N';
        const char TRANS_C = 'C'; // For complex, we will have to used conjugate transpose
        int nnorb = norb * norb;
        int i, j, k, m, n;
        int blk = MIN(((int)(48/norb))*norb, nnorb);
        double complex *buf = malloc(sizeof(double complex) * nnorb*bcount);
        double complex *p1;
        const double complex zalpha = alpha;
        const double complex zbeta  = beta;

        for (n = 0, k = 0; k < bcount; k++) {
                p1 = tbra + (size_t)k * (size_t)nnorb;
                for (i = 0; i < norb; i++) {
                for (j = 0; j < norb; j++, n++) {
                        buf[n] = p1[j*norb+i];
                } }
        }

        for (m = 0; m < nnorb-blk; m+=blk) {
                n = nnorb - m;
                zgemm_(&TRANS_N, &TRANS_C, &blk, &n, &bcount,
                       &zalpha, tket+m, &nnorb, buf+m, &nnorb,
                       &zbeta, rdm2+(size_t)m*nnorb+m, &nnorb);
        }

        n = nnorb - m;
        zgemm_(&TRANS_N, &TRANS_C, &n, &n, &bcount,
               &zalpha, tket+m, &nnorb, buf+m, &nnorb,
               &zbeta, rdm2+(size_t)m*nnorb+m, &nnorb);

        free(buf);
}

double FCIrdm2_a_t1ci_cplx(double complex *ci0, double complex *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci0 += strb_id;
        const int nnorb = norb * norb;
        int i, j, k, a, sign;
        size_t str1;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double complex *pt1, *pci;
        double csum = 0;

        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                pci = ci0 + str1*nstrb;
                pt1 = t1 + i*norb+a;
                if (sign == 0) {
                        break;
                } 
                else if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                pt1[k*nnorb] += pci[k];
                                csum += creal(conj(pci[k]) * pci[k]);
                        }
                } 
                else {
                        for (k = 0; k < bcount; k++) {
                                pt1[k*nnorb] -= pci[k];
                                csum += creal(conj(pci[k]) * pci[k]);
                        }
                }
        }
        return csum;
}


double FCIrdm2_b_t1ci_cplx(double complex *ci0, double complex *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinkb, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        int i, j, a, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double complex *pci = ci0 + (size_t)stra_id * (size_t)nstrb;
        double csum = 0;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        if (sign == 0) {
                                break;
                        } 
                        else {
                                t1[i*norb+a] += ((double)sign) * pci[str1];
                                csum += creal(conj(pci[str1]) * pci[str1]);
                        }
                }
                t1 += nnorb;
                tab += nlinkb;
        }
        return csum;
}

/*
 * Ordinary spin-summed complex 1-/2-RDMs (bra == ket).
 * Build T = (E_alpha + E_beta)|CI> once.  T T^H contains all four
 * spin blocks, avoiding separate aa, bb, and ab driver traversals.
 * The driver applies the same index permutation and Hermitian completion
 * as the spin-resolved kernels; normal-ordering is handled in Python.
 */
void FCIrdm12kern_sf_cplx(double complex *rdm1, double complex *rdm2,
                    double complex *bra, double complex *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_C = 'C';
        const double complex Z1 = 1.0;
        const double D1 = 1.0;
        const int nnorb = norb * norb;
        double csum = 0.0;

        double complex *buf = (double complex*)calloc((size_t)nnorb * (size_t)bcount,
                                                      sizeof(double complex));

        csum = FCIrdm2_a_t1ci_cplx(ket, buf, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);
        csum += FCIrdm2_b_t1ci_cplx(ket, buf, bcount, stra_id, strb_id,
                                  norb, nb, nlinkb, clink_indexb);

        if (csum > CSUMTHR) {
                double complex *v = malloc(sizeof(double complex) * (size_t)bcount);
                for (int kk = 0; kk < bcount; kk++) {
                        v[kk] = conj(bra[(size_t)stra_id*nb + strb_id + kk]);
                }
                zgemv_(&TRANS_N, &nnorb, &bcount, &Z1, buf, &nnorb,
                    v, &INC1, &Z1, rdm1, &INC1);
                free(v);

                switch (symm) {
                case BRAKETSYM:
                        /*
                         * rdm2 += buf * buf^H
                         */
                        zherk_(&UP, &TRANS_N, &nnorb, &bcount,
                               &D1, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;
                // Not tested this yet.
                case PARTICLESYM:
                        tril_particle_symm_cplx(rdm2, buf, buf, bcount, norb, 1.0, 1.0);
                        break;
                default:
                        /*
                         * rdm2 += buf * buf^H
                         */
                        zgemm_(&TRANS_N, &TRANS_C, &nnorb, &nnorb, &bcount,
                               &Z1, buf, &nnorb, buf, &nnorb,
                               &Z1, rdm2, &nnorb);
                }
        }
        free(buf);
}

/*
 * ***********************************************
 *      2pdm kernel for alpha^i alpha_j | ci0 >
 *       (with complex FCI wavefunction)
 * ***********************************************
 */
void FCIrdm12kern_a_cplx(double complex *rdm1, double complex *rdm2,
                    double complex *bra, double complex *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_C = 'C';
        const double complex Z1 = 1.0;
        const double D1 = 1.0;
        const int nnorb = norb * norb;
        double csum = 0.0;

        double complex *buf = (double complex*)calloc((size_t)nnorb * (size_t)bcount,
                                                      sizeof(double complex));

        csum = FCIrdm2_a_t1ci_cplx(ket, buf, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);

        if (csum > CSUMTHR) {
                double complex *v = malloc(sizeof(double complex) * (size_t)bcount);
                for (int kk = 0; kk < bcount; kk++) {
                        v[kk] = conj(bra[stra_id*nb + strb_id + kk]);
                }
                zgemv_(&TRANS_N, &nnorb, &bcount, &Z1, buf, &nnorb,
                    v, &INC1, &Z1, rdm1, &INC1);
                free(v);

                switch (symm) {
                case BRAKETSYM:
                        /*
                         * rdm2 += buf * buf^H
                         */
                        zherk_(&UP, &TRANS_N, &nnorb, &bcount,
                               &D1, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;
                // Not tested this yet.
                case PARTICLESYM:
                        tril_particle_symm_cplx(rdm2, buf, buf, bcount, norb, 1.0, 1.0);
                        break;
                default:
                        /*
                         * rdm2 += buf * buf^H
                         */
                        zgemm_(&TRANS_N, &TRANS_C, &nnorb, &nnorb, &bcount,
                               &Z1, buf, &nnorb, buf, &nnorb,
                               &Z1, rdm2, &nnorb);
                }
        }
        free(buf);
}

/*
 * ***********************************************
 *      2pdm kernel for beta^i beta_j | ci0 >
 *       (with complex FCI wavefunction)
 * ***********************************************
 */
void FCIrdm12kern_b_cplx(double complex *rdm1, double complex *rdm2,
                    double complex *bra, double complex *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_C = 'C';
        const double complex Z1 = 1.0;
        const double D1 = 1.0;
        const int nnorb = norb * norb;
        double csum;
        double complex *buf = calloc((size_t)nnorb * (size_t)bcount, sizeof(double complex));

        csum = FCIrdm2_b_t1ci_cplx(ket, buf, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);
        if (csum > CSUMTHR) {
                double complex *v = malloc(sizeof(double complex) * (size_t)bcount);
                for (int kk = 0; kk < bcount; kk++) {
                        v[kk] = conj(bra[stra_id*nb + strb_id + kk]);
                }
                
                zgemv_(&TRANS_N, &nnorb, &bcount, &Z1, buf, &nnorb,
                    v, &INC1, &Z1, rdm1, &INC1);
                
                free(v);

                switch (symm) {
                case BRAKETSYM:
                        zherk_(&UP, &TRANS_N, &nnorb, &bcount,
                                &D1, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;

                case PARTICLESYM:
                        tril_particle_symm_cplx(rdm2, buf, buf, bcount, norb, 1.0, 1.0);
                        break;
                default:
                        zgemm_(&TRANS_N, &TRANS_C, &nnorb, &nnorb, &bcount,
                                &Z1, buf, &nnorb, buf, &nnorb,
                                &Z1, rdm2, &nnorb);
                }
        }
        free(buf);
}


/*
 * Same-spin complex transition 1- and 2-RDM kernels.  buf0 is the
 * one-body-excited ket and buf1 is the one-body-excited bra, so the 2-TDM
 * contraction is buf0 * buf1^H rather than the ket-ket product used by the
 * ordinary RDM kernels above.
 */
void FCItdm12kern_a_cplx(double complex *tdm1, double complex *tdm2,
                         double complex *bra, double complex *ket,
                         int bcount, int stra_id, int strb_id,
                         int norb, int na, int nb, int nlinka, int nlinkb,
                         _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_C = 'C';
        const double complex Z1 = 1.0;
        const int nnorb = norb * norb;
        double csum;
        double complex *buf0 = calloc((size_t)nnorb * (size_t)bcount,
                                      sizeof(double complex));
        double complex *buf1 = calloc((size_t)nnorb * (size_t)bcount,
                                      sizeof(double complex));
        double complex *v = malloc(sizeof(double complex) * (size_t)bcount);

        (void)symm;
        csum = FCIrdm2_a_t1ci_cplx(bra, buf1, bcount, stra_id, strb_id,
                                   norb, nb, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end_a; }
        csum = FCIrdm2_a_t1ci_cplx(ket, buf0, bcount, stra_id, strb_id,
                                   norb, nb, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end_a; }

        for (int k = 0; k < bcount; k++) {
                v[k] = conj(bra[(size_t)stra_id * (size_t)nb + strb_id + k]);
        }
        zgemv_(&TRANS_N, &nnorb, &bcount, &Z1, buf0, &nnorb,
               v, &INC1, &Z1, tdm1, &INC1);
        zgemm_(&TRANS_N, &TRANS_C, &nnorb, &nnorb, &bcount,
               &Z1, buf0, &nnorb, buf1, &nnorb, &Z1, tdm2, &nnorb);

_normal_end_a:
        free(v);
        free(buf1);
        free(buf0);
}


void FCItdm12kern_b_cplx(double complex *tdm1, double complex *tdm2,
                         double complex *bra, double complex *ket,
                         int bcount, int stra_id, int strb_id,
                         int norb, int na, int nb, int nlinka, int nlinkb,
                         _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_C = 'C';
        const double complex Z1 = 1.0;
        const int nnorb = norb * norb;
        double csum;
        double complex *buf0 = calloc((size_t)nnorb * (size_t)bcount,
                                      sizeof(double complex));
        double complex *buf1 = calloc((size_t)nnorb * (size_t)bcount,
                                      sizeof(double complex));
        double complex *v = malloc(sizeof(double complex) * (size_t)bcount);

        (void)symm;
        csum = FCIrdm2_b_t1ci_cplx(bra, buf1, bcount, stra_id, strb_id,
                                   norb, nb, nlinkb, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end_b; }
        csum = FCIrdm2_b_t1ci_cplx(ket, buf0, bcount, stra_id, strb_id,
                                   norb, nb, nlinkb, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end_b; }

        for (int k = 0; k < bcount; k++) {
                v[k] = conj(bra[(size_t)stra_id * (size_t)nb + strb_id + k]);
        }
        zgemv_(&TRANS_N, &nnorb, &bcount, &Z1, buf0, &nnorb,
               v, &INC1, &Z1, tdm1, &INC1);
        zgemm_(&TRANS_N, &TRANS_C, &nnorb, &nnorb, &bcount,
               &Z1, buf0, &nnorb, buf1, &nnorb, &Z1, tdm2, &nnorb);

_normal_end_b:
        free(v);
        free(buf1);
        free(buf0);
}

/*
 * ***********************************************
 *      2pdm kernel for alpha^i beta_j | ci0 >
 *       (with complex FCI wavefunction)
 * ***********************************************
 */
void FCItdm12kern_ab_cplx(double complex *tdm1, double complex *tdm2,
                     double complex *bra, double complex *ket,
                     int bcount, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const char TRANS_N = 'N';
        const char TRANS_C = 'C';
        const double complex Z1 = 1.0;
        const int nnorb = norb * norb;
        double csum;
        double complex *bufb = calloc((size_t)nnorb * (size_t)bcount, sizeof(double complex));
        double complex *bufa = calloc((size_t)nnorb * (size_t)bcount, sizeof(double complex));

        csum = FCIrdm2_a_t1ci_cplx(bra, bufa, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end; }

        csum = FCIrdm2_b_t1ci_cplx(ket, bufb, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end; }

        zgemm_(&TRANS_N, &TRANS_C, &nnorb, &nnorb, &bcount,
               &Z1, bufb, &nnorb, bufa, &nnorb, &Z1, tdm2, &nnorb);

_normal_end:
        free(bufb);
        free(bufa);
}
