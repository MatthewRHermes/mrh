#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include "fblas.h"

void SDFKmat (double * sparse_cderi, double * dense_dm, double * dense_vk, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    int npp = npair * (npair + 1) / 2;

#pragma omp parallel default(shared)
{

    int ix_pair1 = 0;
    int ix_pair2 = 0;
    int pair1, pair2, ipp, i1ao, i2ao, j1ao, j2ao, vk_ptr, dm_ptr; 
    double eri;
    pair1 = nonzero_pair[ix_pair1];
    i1ao = tril_iao[pair1];
    j1ao = tril_jao[pair1];
    pair2 = nonzero_pair[ix_pair2];
    i2ao = tril_iao[pair2];
    j2ao = tril_jao[pair2];

#pragma omp for schedule(static) 

    for (ipp = 0; ipp < npp; ipp++){

        // Dot product over auxbasis
        eri = ddot_(&naux, sparse_cderi + ix_pair1, &npair, sparse_cderi + ix_pair2, &npair);

        // Permutations from lower-triangular form
        vk_ptr = (i1ao*nao) + i2ao;
        dm_ptr = (j1ao*nao) + j2ao;
        dense_vk[vk_ptr] += eri * dense_dm[dm_ptr];
        dense_vk[dm_ptr] += eri * dense_dm[vk_ptr];
        vk_ptr = (j1ao*nao) + i2ao;
        dm_ptr = (i1ao*nao) + j2ao;
        dense_vk[vk_ptr] += eri * dense_dm[dm_ptr];
        dense_vk[dm_ptr] += eri * dense_dm[vk_ptr];

        // Pair-of-pair indexing
        ix_pair2++;
        if (ix_pair2 > ix_pair1){
            ix_pair1++;
            pair1 = nonzero_pair[ix_pair1];
            i1ao = tril_iao[pair1];
            j1ao = tril_jao[pair1];
            ix_pair2 = 0;
        }
        pair2 = nonzero_pair[ix_pair2];
        i2ao = tril_iao[pair2];
        j2ao = tril_jao[pair2];
    }

}
}

