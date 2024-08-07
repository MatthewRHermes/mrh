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

void LASSIRDMdputSD (double * SDsum, double * SDterm, int SDlen,
                     double * sivec, int sivec_nbas, int sivec_nroots,
                     long * bra, long * ket, double * wgt, int nelem)
{
    const unsigned int i_one = 1;

    double fac = 0;
    double * sicol = sivec;
    double * SDtarget = SDsum;

    for (int iroot = 0; iroot < sivec_nroots; iroot++){
        sicol = sivec + (iroot*sivec_nbas);
        SDtarget = SDsum + (iroot*SDlen);

        fac = 0;

        #pragma omp parallel for schedule(static) reduction(+:fac)
        for (int ielem = 0; ielem < nelem; ielem++){
            fac += sicol[bra[ielem]] * sicol[ket[ielem]] * wgt[ielem];
        }

        //daxpy_(&SDlen, &fac, SDterm, &i_one, SDtarget, &i_one);
        #pragma omp parallel
        {
            int nblk = omp_get_num_threads ();
            nblk = (SDlen+nblk-1) / nblk;
            int toff = nblk * omp_get_thread_num ();
            nblk = MIN (SDlen, toff+nblk);
            nblk = nblk - toff;
            daxpy_(&nblk, &fac, SDterm+toff, &i_one, SDtarget+toff, &i_one);
        }
    }

}
