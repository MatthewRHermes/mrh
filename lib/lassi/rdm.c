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
                     int nroots, int nelem_dest,
                     int * SDdest_idx, int * SDsrc_idx, int * SDlen,
                     int nidx)
{
const unsigned int i_one = 1;
#pragma omp parallel
{
    double * mySDsrc;
    double * mySDdest;
    #pragma omp for 
    for (int iidx = 0; iidx < nidx; iidx++){
        mySDsrc = SDsrc + SDsrc_idx[iidx];
        mySDdest = SDdest + SDdest_idx[iidx];
        for (int iroot = 0; iroot < nroots; iroot++){
            daxpy_(SDlen+iidx, fac+iroot,
                   mySDsrc, &i_one,
                   mySDdest+iroot*nelem_dest, &i_one);
        }
    }

}
}

void LASSIRDMdputSD (double * SDdest, double * SDsrc,
                     int nroots, int nelem_dest, int nelem_src,
                     int * SDdest_idx, int * SDsrc_idx, int * SDlen,
                     int nidx)
{
const unsigned int i_one = 1;
const double d_one = 1.0;
#pragma omp parallel
{
    double * mySDsrc;
    double * mySDdest;
    #pragma omp for 
    for (int iidx = 0; iidx < nidx; iidx++){
        mySDsrc = SDsrc + SDsrc_idx[iidx];
        mySDdest = SDdest + SDdest_idx[iidx];
        for (int iroot = 0; iroot < nroots; iroot++){
            daxpy_(SDlen+iidx, &d_one,
                   mySDsrc+iroot*nelem_src, &i_one,
                   mySDdest+iroot*nelem_dest, &i_one);
        }
    }

}
}
