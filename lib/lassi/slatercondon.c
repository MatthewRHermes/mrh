#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <complex.h>
#include "../fblas.h"

#ifndef MINMAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MINMAX
#endif

bool is_interaction_coupled (int * nelec_fs_bra, int * nelec_fs_ket, const int nfrags)
{
    int dna, dnb;
    int sa = 0;
    int sb = 0;
    int d = 0;
    for (int ifrag = 0; ifrag < nfrags; ifrag++){
        dna = nelec_fs_bra[ifrag*2] - nelec_fs_ket[ifrag*2];
        dnb = nelec_fs_bra[(ifrag*2)+1] - nelec_fs_ket[(ifrag*2)+1];
        sa += dna;
        sb += dnb;
        d += abs (dna);
        d += abs (dnb);
    }
    bool is_coupled = (sa == 0) && (sb == 0) && (d <= 4);
    return is_coupled;
}

long SCcntinter (int * nelec_rfs_bra, int * nelec_rfs_ket,
                 const long nroots_bra, const long nroots_ket,
                 const int nfrags)
{
/* Count the number of valid interactions between model states based on the electron numbers.
   Input:
        nelec_rfs_bra : array of shape (nroots_bra,nfrags,2)
        nelec_rfs_ket : array of shape (nroots_ket,nfrags,2)

   Returns:
        n : Number of pairs of model states which can be coupled by the Hamiltonian.
*/
long n = 0;
const long nroots2 = nroots_bra * nroots_ket;
const long rstride = nfrags * 2;
#pragma omp parallel
{
    long iket, ibra;
    long my_n = 0;
    int * nelec_fs_bra;
    int * nelec_fs_ket;
    #pragma omp for schedule(static)
    for (long iel = 0; iel < nroots2; iel++){
        iket = iel % nroots_ket;
        ibra = iel / nroots_ket;
        nelec_fs_ket = nelec_rfs_ket + (iket * rstride);
        nelec_fs_bra = nelec_rfs_bra + (ibra * rstride);
        if (is_interaction_coupled (nelec_fs_bra, nelec_fs_ket, nfrags)){
            my_n++;
        }
    }
    #pragma omp critical
    {
        n += my_n;
    }
}
return n;
}

void SClistinter (long * exc, int * nelec_rfs_bra, int * nelec_rfs_ket,
                  const long nexc, const long nroots_bra, const long nroots_ket,
                  const int nfrags)
{
/* List all valid interactions between model states based on the electron numbers
   Input:
        nelec_rfs_bra : array of shape (nroots_bra,nfrags,2)
        nelec_rfs_ket : array of shape (nroots_ket,nfrags,2)

   Output:
        exc : array of shape (nexc,2)
*/
const long nroots2 = nroots_bra * nroots_ket;
const long rstride = nfrags * 2;
const long one = 1;
const long two = 2;
long iexc = 0;
#pragma omp parallel
{
    int nt = omp_get_num_threads ();
    int it = omp_get_thread_num ();
    long * my_exc = malloc (nexc*two*sizeof(long));
    long my_iexc = 0;
    long iket, ibra;
    int * nelec_fs_bra;
    int * nelec_fs_ket;
    #pragma omp for schedule(static)
    for (long iel = 0; iel < nroots2; iel++){
        iket = iel % nroots_ket;
        ibra = iel / nroots_ket;
        nelec_fs_ket = nelec_rfs_ket + (iket * rstride);
        nelec_fs_bra = nelec_rfs_bra + (ibra * rstride);
        if (is_interaction_coupled (nelec_fs_bra, nelec_fs_ket, nfrags)){
            my_exc[my_iexc*two] = ibra;
            my_exc[(my_iexc*two)+one] = iket;
            my_iexc++;
        }
    }
    #pragma omp critical
    {
        for (long i=0; i<my_iexc*two; i++){
            exc[iexc+i] = my_exc[i];
        }
        iexc += my_iexc*two;
    }
    free (my_exc);
}
}

