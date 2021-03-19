#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include "fblas.h"

#ifndef MINMAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MINMAX
#endif

/* Fock-Space Unitary Coupled Cluster functions
   Cf. Eqs (1)-(12) of JCTC 17 841 2021
   (DOI:10.1021/acs.jctc.0c01052) */

void FSUCCcontract1 (unsigned int * aidx, unsigned int * iidx, double tamp,
    double * psi, unsigned int norb, unsigned int nexc);
{
    /* Evaluate U|Psi> = e^(t a0'a1'...i1i0 - i0'i1'...a1a0) |Psi> 
       The normal-order convention is
       i0 < i1 < i2 < ...
       a0 < a1 < a2 < ...
       Pro tip: add pi/2 to the amplitude to evaluate dU/dt |Psi>

       Input:
            aidx : array of shape (nexc); identifies +cr,-an ops
            iidx : array of shape (nexc); identifies +an,-cr ops
                Note: creation operators are applied left < right;
                annihilation operators are applied right < left
            tamp : the amplitude or angle

       Input/Output:
            psi : array of shape (2**norb); contains wfn
                Modified in place. Make a copy in the caller
                if you don't want to modify the input
    */

    int int_one = 1;
    uint64_t ndet = (1<<(norb-nexc)); // 2**(norb-nexc)
    double ct = cos (tamp); // (ct -st) (ia) -> (ia)
    double st = sin (tamp); // (st  ct) (ai) -> (ai)

#pragma omp parallel default(shared)
{

    uint64_t det, det_00, det_ia, det_ai;
    unsigned int p, q, sgnbit;
    int sgn;
    double cia, cai;

#pragma omp for schedule(static)

    for (det = 0; det < ndet; det++){
        // "det" here is the string of uninvolved spinorbitals
        // To find the full det string I have to insert i, a
        det_00 = det;
        det_ia = 0;
        for (p = 0; p < nexc; p++){
            det_ia |= ((1<<iidx[p]) | (1<<aidx[p]));
        } // det_ia now identifies any of either i or a
        for (p = 0; p < MAX (iidx[nexc-1], aidx[0]);  p++){
            if (det_ia & (1<<p)){
                det_ai  = det_00 >> p;     // lr  -> l
                det_00 &= ((1<<p)-1);      // lr  -> r
                det_00 |= det_ai << (p+1); // l,r -> l0r
            } 
        }
        // det_00 now has unoccupied orbitals in the i,a positions
        // Next, make strings describing i,a orbitals occupied in their
        // proper positions
        det_ia = 0;
        det_ai = 0;
        for (p = 0; p < nexc; p++){
            det_ia |= (1<<iidx[p]); // i is occupied
            det_ai |= (1<<aidx[p]); // a is occupied
        }
        det_ia |= det_00;
        det_ai |= det_00;
        // The sign for the whole excitation is the product of the
        // sign incurred by doing this to det_ia
        // ...i2'...i1'...i0'|0> -> i0'i1'i2'...|0>
        // and doing this to det_ai
        // ...a2'...a1'...a0'|0> -> a0'a1'a2'...|0>
        sgnbit = 0;
        for (p = 0; p < nexc; p++){
            for (q = iidx[p]+1; q < norb; q++){
                sgnbit ^= (1 & (det>>q));
            }
            for (q = aidx[p]+1; q < norb; q++){
                sgnbit ^= (1 & (det>>q));
            }
        }
        sgn = int_one - 2*((int) sgnbit);
        // The rest of the math is trivial
        cia = sgn * psi[det_ia];
        cai = sgn * psi[det_ai];
        psi[det_ia] = (ct*cia) - (st*cia);
        psi[det_ai] = (st*cia) + (ct*cia);
    }

}

}

