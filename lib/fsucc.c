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

#ifndef PI
#define PI 3.14159265358979323846
#endif

/* Fock-Space Unitary Coupled Cluster functions
   Cf. Eqs (1)-(12) of JCTC 17 841 2021
   (DOI:10.1021/acs.jctc.0c01052) 

   No spin, no symmetry, no derivatives, not even any looping over amplitudes.
   Do all of that in the Python caller because none of it can easily be
   multithreaded.

   TODO: double-check that this actually works for number-symmetry-breaking
   generators (it should work for 1P or 1H operators, but I haven't checked 
   nPmH in general
*/

void FSUCCcontract1 (uint8_t * aidx, uint8_t * iidx, double tamp,
    double * psi, unsigned int norb, unsigned int na, unsigned int ni)
{
    /* Evaluate U|Psi> = e^(t a0'a1'...i1i0 - i0'i1'...a1a0) |Psi> 
       Pro tip: add pi/2 to the amplitude to evaluate dU/dt |Psi>

       Input:
            aidx : array of shape (na); identifies +cr,-an ops
            iidx : array of shape (ni); identifies +an,-cr ops
                Note: creation operators are applied left < right;
                annihilation operators are applied right < left
            tamp : the amplitude or angle

       Input/Output:
            psi : array of shape (2**norb); contains wfn
                Modified in place. Make a copy in the caller
                if you don't want to modify the input
    */

    const int int_one = 1;
    const double ct = cos (tamp); // (ct -st) (ia) -> (ia)
    const double st = sin (tamp); // (st  ct) (ai) -> (ai)
    int r;
    uint64_t det_i = 0; // i is occupied
    for (r = 0; r < ni; r++){ det_i |= (1<<iidx[r]); }
    uint64_t det_a = 0; // a is occupied
    for (r = 0; r < na; r++){ det_a |= (1<<aidx[r]); }
    // all other spinorbitals in det_i, det_a unoccupied
    uint64_t ndet = (1<<norb); // 2**norb
    for (r = 0; r < norb; r++){ if ((det_i|det_a) & (1<<r)){
        ndet >>= 1; // pop 1 spinorbital per unique i,a
        // we only sum over the spectator-spinorbital determinants
    }}

#pragma omp parallel default(shared)
{

    uint64_t det, det_00, det_ia, det_ai;
    unsigned int p, q, sgnbit;
    int sgn;
    double cia, cai;

#pragma omp for schedule(static)

    for (det = 0; det < ndet; det++){
        // "det" here is the string of spectator spinorbitals
        // To find the full det string I have to insert i, a in ascending order
        det_00 = det;
        for (p = 0; p < norb; p++){
            if ((det_i|det_a) & (1<<p)){
                det_00 = (((det_00 >> p) << (p+1)) // move left bits 1 left
                         | (det_00 & ((1<<p)-1))); // keep right bits
            } 
        } // det_00: spectator spinorbitals; all i, a bits unset
        det_ia = det_00 | det_i;
        det_ai = det_00 | det_a;
        // The sign for the whole excitation is the product of the sign incurred
        // by doing this to det_ia:
        // ...i2'...i1'...i0'|0> -> i0'i1'i2'...|0>
        // and doing this to det_ai:
        // ...a2'...a1'...a0'|0> -> a0'a1'a2'...|0>.
        // To implement this without assuming normal-ordered generators
        // (i.e., i0 < i1 < i2 or a0 < a1 < a2)
        // we need to pop creation operators from the string in the order that
        // we move them to the front. Repurpose det_00 for this.
        sgnbit = 0; // careful to only modify the first bit of this
        det_00 = det_ia;
        for (p = 0; p < ni; p++){
            for (q = iidx[p]+1; q < norb; q++){
                sgnbit ^= (det_00 & (1<<q))>>q; // c1'c2' = -c2'c1' sign toggle
            }
            det_00 ^= (1<<iidx[p]); // pop ip
        }
        det_00 = det_ai;
        for (p = 0; p < na; p++){
            for (q = aidx[p]+1; q < norb; q++){
                sgnbit ^= (det_00 & (1<<q))>>q; // c1'c2' = c2'c1' sign toggle
            }
            det_00 ^= (1<<aidx[p]); // pop ap
        }
        sgn = int_one - 2*((int) sgnbit);
        // The rest of the math is trivial
        cia = sgn * psi[det_ia];
        cai = sgn * psi[det_ai];
        psi[det_ia] = (ct*cia) - (st*cai);
        psi[det_ai] = (st*cia) + (ct*cai);
    }

}

}

