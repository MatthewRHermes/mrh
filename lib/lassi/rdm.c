#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include "../fblas.h"

#ifndef MINMAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MINMAX
#endif

/*
    # A C version of the below would need:
    #   all args of _put_SD?_ 
    #   self.si, in some definite order
    #   length of _put_SD?_ args, ncas, nroots_si, maybe nstates?
    # If I wanted to index down, I would also need
    #   ncas_sub, nfrags, inv, len (inv)

    def _put_SD1_(self, bra, ket, D1, wgt):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        si_dm = self.si[bra,:] * self.si[ket,:].conj ()
        fac = np.dot (wgt, si_dm)
        self.rdm1s[:] += np.multiply.outer (fac, D1)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw
        
    def _put_SD2_(self, bra, ket, D2, wgt):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        si_dm = self.si[bra,:] * self.si[ket,:].conj ()
        fac = np.dot (wgt, si_dm)
        self.rdm2s[:] += np.multiply.outer (fac, D2)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw
*/

void LASSIRDMdgetwgtfac (double * fac, double * wgt, double * sivec,
                         int nbas, int nroots,
                         long * bra, long * ket, int nelem)
{
/* Evaluate fac = np.dot (si[bra,:]*si[ket,:], wgt) with multiple threads

   Input:
        wgt : array of shape (nelem); contains overlap & permutation factors
        sivec : array of shape (nroots,nbas); contains SI vector coefficients
        bra : array of shape (nelem); identifies SI rows
        ket : array of shape (nelem); identifies SI rows

   Output:
        fac : array of shape (nroots)
*/
#pragma omp parallel
{
    double * sicol;
    long bra_i, ket_i;
    double * my_fac = malloc (nroots*sizeof(double));
    for (int iroot = 0; iroot < nroots; iroot++){ my_fac[iroot] = 0; }
    #pragma omp for schedule(static)
    for (int ielem = 0; ielem < nelem; ielem++){
        for (int iroot = 0; iroot < nroots; iroot++){
            sicol = sivec + (iroot*nbas);
            bra_i = bra[ielem];
            ket_i = ket[ielem];
            my_fac[iroot] += sicol[bra_i] * sicol[ket_i] * wgt[ielem];
        }
    }
    #pragma omp critical
    {
        for (int iroot=0; iroot < nroots; iroot++){
            fac[iroot] += my_fac[iroot];
        }
    }
    free (my_fac);
}
}

void LASSIRDMdsumSD (double * SDdest, double * fac, double * SDsrc,
                     int nroots, int nelem_dest)
{
/* Evaluate SDdest = np.multiply.outer (fac, SDsrc) with multiple threads.

   Input:
        fac : array of shape (nroots)
        SDsrc : array of shape (nelem_dest)

   Output:
        SDdest : array of shape (nroots,nelem_dest)
*/
const unsigned int i_one = 1;
#pragma omp parallel
{
    int nt = omp_get_num_threads ();
    int it = omp_get_thread_num ();
    int nel0 = nelem_dest / nt;
    int nel1 = nelem_dest % nt;
    int tstart = (nel0*it) + MIN(nel1,it);
    int tlen = it<nel1 ? nel0+1 : nel0;
    double * mySDsrc = SDsrc+tstart;
    double * mySDdest = SDdest+tstart;
    for (int iroot = 0; iroot < nroots; iroot++){
        daxpy_(&(tlen), fac+iroot,
               mySDsrc, &i_one,
               mySDdest+iroot*nelem_dest, &i_one);
    }
}
}

void LASSIRDMdputSD1 (double * SDdest, double * SDsrc,
                      int nroots, int ndest, int nsrc,
                      int * SDdest_idx, int * SDsrc_idx, int * SDlen,
                      int nidx)
{
/* Add to segmented elements of RDM1s arrays from a contiguous source array

   Input:
        SDsrc : array of shape (nroots,2,nsrc*nsrc)
        SDdest_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDdest
        SDsrc_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDsrc
        SDlen : array of shape (nidx); lengths of contiguous blocks

   Input/Output:
        SDdest : array of shape (nroots,2,ndest*ndest)
            Elements of SDsrc corresponding to SDdest_idx are added
*/
const unsigned int i_one = 1;
const double d_one = 1.0;
const int npdest = ndest*ndest;
const int npsrc = nsrc*nsrc;
const int nblk = nroots*2*nidx;
const int lroot = 2*nidx;
const int SDsrc_size = nroots*2*npsrc;
const int SDdest_size = nroots*2*npdest;
#pragma omp parallel
{
    int iroot, ispin, iidx, j;
    int sidx, didx;
    #pragma omp for 
    for (int i = 0; i < nblk; i++){
        iroot = i/lroot;
        j = i%lroot;
        ispin = j/nidx;
        iidx = j%nidx;
        assert (iroot < nroots);
        assert (ispin < 2);
        assert (iidx < nidx);
        sidx = (iroot*2 + ispin)*npsrc + SDsrc_idx[iidx];
        didx = (iroot*2 + ispin)*npdest + SDdest_idx[iidx];
        assert (sidx < SDsrc_size);
        assert (didx < SDdest_size);
        daxpy_(SDlen+iidx, &d_one, SDsrc+sidx, &i_one, SDdest+didx, &i_one);
    }
}
}

void LASSIRDMdputSD2 (double * SDdest, double * SDsrc,
                      int nroots, int ndest, int nsrc, int * pdest,
                      int * SDdest_idx, int * SDsrc_idx, int * SDlen,
                      int nidx)
{
/* Add to segmented elements of RDM2s arrays from a contiguous source array

   Input:
        SDsrc : array of shape (nroots,4,nsrc*nsrc,nsrc*nsrc)
        pdest : array of shape (nsrc,nsrc)
            Indices of all addressed elements in the second-minor dimension of SDdest
        SDdest_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDdest
        SDsrc_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDsrc
        SDlen : array of shape (nidx); lengths of contiguous blocks

   Input/Output:
        SDdest : array of shape (nroots,2,ndest*ndest,ndest*ndest)
            Elements of SDsrc corresponding to SDdest_idx are added
*/
const unsigned int i_one = 1;
const double d_one = 1.0;
const int npdest = ndest*ndest;
const int npsrc = nsrc*nsrc;
const int ntdest = npdest*npdest;
const int ntsrc = npsrc*npsrc;
const int nblk = nroots*4*npsrc*nidx;
const int lroot = 4*npsrc*nidx;
const int lspin = npsrc*nidx;
const int SDsrc_size = nroots*4*ntsrc;
const int SDdest_size = nroots*4*ntdest;
#pragma omp parallel
{
    int ipdest, iroot, ispin, iidx, j;
    int sidx, didx;
    #pragma omp for 
    for (int i = 0; i < nblk; i++){
        iroot = i/lroot;
        j = i%lroot;
        ispin = j/lspin;
        j = j%lspin;
        ipdest = j/nidx;
        iidx = j%nidx;
        assert (iroot < nroots);
        assert (ispin < 4);
        assert (ipdest < npsrc);
        assert (iidx < nidx);
        sidx = (iroot*4 + ispin)*ntsrc + ipdest*npsrc + SDsrc_idx[iidx];
        didx = (iroot*4 + ispin)*ntdest + pdest[ipdest]*npdest + SDdest_idx[iidx];
        assert (sidx < SDsrc_size);
        assert (didx < SDdest_size);
        daxpy_(SDlen+iidx, &d_one, SDsrc+sidx, &i_one, SDdest+didx, &i_one);
    }
}
}
