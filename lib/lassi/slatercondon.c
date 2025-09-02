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

// http://www.isthe.com/chongo/tech/comp/fnv/index.html#FNV-1a
// ChatGPT also helped me find this...
uint64_t fnv_1a (long * arr, size_t len) {
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    uint64_t prime = 1099511628211ULL;       // FNV prime

    for (size_t i = 0; i < len; i++){
        hash ^= (uint64_t) arr[i];
        hash *= prime;
    }

    return hash;
}

void bubblesort (long * a, size_t len_a)
{
    long temp;
    for (size_t i = 0; i < (len_a-1); i++){
        for (size_t j = 0; j < (len_a-i-1); j++){
            if (a[j] > a[j+1]){
                temp = a[j];
                a[j] = a[j+1];
                a[j+1] = temp;
            }
        }
    }
}

void SCfprint (uint64_t * fprint, uint64_t * fprintLT, long * exc, long * urootstr,
               const int nfrags, const int exc_nrows, const int exc_ncols,
               const int urootstr_ncols)
{
const long three = 3;
#pragma omp parallel
{
    int nt = omp_get_num_threads ();
    int it = omp_get_thread_num ();
    long * fprint_row = malloc (three*nfrags*sizeof(long));
    long * brastr = fprint_row + nfrags;
    long * ketstr = brastr + nfrags;
    long bra, ket;
    bool trans = false;
    bool unsorted = true;
    long * exc_row;
    long ifrag;
    int lbra, lket;
    #pragma omp for schedule(static)
    for (size_t i = 0; i < exc_nrows; i++){
        exc_row = exc + i*exc_ncols;
        bra = exc_row[0];
        ket = exc_row[1];
        for (int j = 0; j < nfrags; j++){
            ifrag = exc_row[j+2];
            fprint_row[j] = ifrag;
        }
        bubblesort (fprint_row, nfrags);
        trans = false;
        unsorted = true;
        for (int j = 0; j < nfrags; j++){
            ifrag = fprint_row[j];
            // Double-check data orientation of urootstr
            brastr[j] = urootstr[(bra*urootstr_ncols) + ifrag];
            ketstr[j] = urootstr[(ket*urootstr_ncols) + ifrag];
            if (unsorted && (brastr[j] < ketstr[j])){ trans = true; }
            if (brastr[j] != ketstr[j]){ unsorted = false; }
        }
        fprint[i] = fnv_1a (fprint_row, three*nfrags);
        if (trans){
            for (int j = 0; j < nfrags; j++){
                ifrag = brastr[j];
                brastr[j] = ketstr[j];
                ketstr[j] = ifrag;
            }
        }
        fprintLT[i] = fnv_1a (fprint_row, three*nfrags);
    }
    free (fprint_row);
}
}



