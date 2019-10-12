#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include "fblas.h"

void SINTKmatpermR (double eri, double * dense_vk, double * dense_dm, unsigned int iao, unsigned int jao, unsigned int kao, unsigned int lao, unsigned int nao)
{
    dense_vk[(iao*nao) + kao] += eri * dense_dm[(jao*nao) + lao];
}
void SDFKmatR_obsolete (double * sparse_cderi, double * dense_dm, double * dense_vk, int * nonzero_pair, int * tril_pair1, int * tril_pair2, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{
    /* Get exchange matrix from a sparse cderi with pairs indicated by nonzero_pair. tril_* are lower-triangular index lists.
       This is prone to huge rounding errors if I reduce it on the fly, so each thread should keep its own vk matrix, and the final reduction
       should be carried out post facto.
       sparse_cderi has the auxiliary basis on the faster-moving index (so the increment of the dot product is 1 and the offset to the address has a factor of naux in it)
       This is the worst implementation possible; never use this. I'm keeping it here only for reference.
    */

    unsigned int npp = npair * (npair + 1) / 2;
    const int one = 1;

#pragma omp parallel default(shared)
{

    unsigned int ithread = omp_get_thread_num (); // Thread index, used in final reduction
    unsigned int ipp; // Pair-of-pair index
    unsigned int ix_pair1, ix_pair2; // Pair index
    unsigned int pair1, pair2; // Pair identity
    unsigned int iao, jao, kao, lao; // AO indices
    double eri;
    double * my_vk = dense_vk + (nao*nao*ithread);

#pragma omp for schedule(static) 

    for (ipp = 0; ipp < npp; ipp++){

        // Pair-of-pair indexing
        ix_pair1 = tril_pair1[ipp];
        pair1 = nonzero_pair[ix_pair1];
        iao = tril_iao[pair1];
        jao = tril_jao[pair1];
        ix_pair2 = tril_pair2[ipp];
        pair2 = nonzero_pair[ix_pair2];
        kao = tril_iao[pair2];
        lao = tril_jao[pair2];

        // Dot product over auxbasis
        eri = ddot_(&naux, sparse_cderi + (ix_pair1*naux), &one, sparse_cderi + (ix_pair2*naux), &one);
        if (iao == jao){ eri /= 2; }
        if (kao == lao){ eri /= 2; }
        if (pair1 == pair2){ eri /= 2; }

        // (ij|kl) external index permutations: i <-> j, k <-> l, and ij <-> kl
        SINTKmatpermR (eri, my_vk, dense_dm, iao, jao, kao, lao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, jao, iao, kao, lao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, iao, jao, lao, kao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, jao, iao, lao, kao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, kao, lao, iao, jao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, lao, kao, iao, jao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, kao, lao, jao, iao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, lao, kao, jao, iao, nao);

    }

}

}


void SDFKmatR1 (double * sparse_cderi, double * dense_dm, double * dense_int, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    /* 
    Do the first contraction in the calculation of the exchange matrix using a sparse CDERI array and a density matrix:
    h(i,j,P) = (ik|P) D^k_j
    
    Input:
        sparse_cderi : array of shape (npair, naux); contains CDERIs
        dense_dm : array of shape (nao, nao); contains density matrix
        nonzero_pair : array of shape (npair) ; contains addresses of nao pairs with nonzero overlap
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        npair : number of pairs with nonzero overlap ; <= nao * (nao+1) / 2 ; propto nao in thermodynamic limit
        nao : number of AOs
        naux : number of auxiliary basis functions

    Output:
        dense_int : array of shape (nao, nao, naux); contains inner product of density matrix with CDERIs
           The first (slowest-changing) AO index is the CDERI index; the second is the density matrix index.
           This is necessary in order to use dger. I'll want to transpose this before doing the second contraction
    */

    const unsigned int i_one = 1;
    const unsigned int nao_naux = nao * naux;
    const double d_one = 1.0;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, pair_id; // Pair index and identity
    unsigned int iao, jao; // AO indices
    // Array pointers for lapack
    double * out_ptr; 
    double * cderi_ptr;
    double * dm_ptr;

#pragma omp for schedule(static) 

    for (ipair = 0; ipair < npair; ipair++){
        cderi_ptr = sparse_cderi + (ipair * naux); // This points to a vector of length naux
        pair_id = nonzero_pair[ipair]; 
        iao = tril_iao[pair_id];
        jao = tril_jao[pair_id];
        if (iao % nthreads == ithread){ // Prevent race condition by assigning each CDERI output idx to 1 thread (output is NOT lower-triangular!)
            out_ptr = dense_int + (iao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dm_ptr = dense_dm + (jao * nao); // This points to a vector of length nao
            dger_(&naux, &nao, &d_one, cderi_ptr, &i_one, dm_ptr, &i_one, out_ptr, &naux);
            /* ^ I may have this backwards but if dger_ interprets everything in col-major form, this is what I need to get a
            row-major output of the type I want. Likewise below! */
        }
        if (iao != jao){ if (jao % nthreads == ithread){ 
            out_ptr = dense_int + (jao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dm_ptr = dense_dm + (iao * nao); // This points to a vector of length nao
            dger_(&naux, &nao, &d_one, cderi_ptr, &i_one, dm_ptr, &i_one, out_ptr, &naux);
        }}
    }

#pragma omp barrier

}
}

void SDFKmatRT (double * arr, double * wrk, int * tril_iao, int * tril_jao, int nao, int naux)
{

    /*
    Transpose the (dense, massive) intermediate from SDFKmatR1 in-place in preparation for the dgemv in the second step

    Input:
        wrk : array of shape (nthreads, naux); used to hold vectors during the transpose
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        nao : number of AOs
        naux : number of auxiliary basis functions

    Input/Output:
        arr : array of shape (nao, nao, naux); contains the output of SDFKmatR1
            On entry, slowest-moving index is the CDERI index
            On exit, slowest-moving index is the density-matrix index
    */

    const unsigned int i_one = 1;
    const unsigned int nao_naux = nao * naux;
    const unsigned int npair = nao * (nao + 1) / 2;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, iao, jao; // AO indices
    // Array pointers for lapack
    double * my_wrk = wrk + (ithread * naux);
    double * vec_ij;
    double * vec_ji;

    for (ipair = 0; ipair < npair; ipair++){ // Race conditions are impossible because the pairs don't talk to each other
        iao = tril_iao[ipair];
        jao = tril_jao[ipair];
        if (iao > jao){ 
            vec_ij = arr + (iao * nao_naux) + (jao * naux);
            vec_ji = arr + (jao * nao_naux) + (iao * naux);
            dcopy_(&naux, vec_ij, &i_one, my_wrk, &i_one);
            dcopy_(&naux, vec_ji, &i_one, vec_ij, &i_one);
            dcopy_(&naux, my_wrk, &i_one, vec_ji, &i_one);
        }
    } 

}
}

void SDFKmatR2 (double * sparse_cderi, double * dense_int, double * dense_vk, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    /* 
    Do the second contraction in the calculation of the exchange matrix using a sparse CDERI array and a density matrix:
    K^i_j = (ik|P) h(j,k,P)
    
    Input:
        sparse_cderi : array of shape (npair, naux); contains CDERIs
        dense_int : array of shape (nao, nao, naux); contains contraction of CDERI with density matrix
            Slowest-moving index is the density matrix index
        nonzero_pair : array of shape (npair) ; contains addresses of nao pairs with nonzero overlap
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        npair : number of pairs with nonzero overlap ; <= nao * (nao+1) / 2 ; propto nao in thermodynamic limit
        nao : number of AOs
        naux : number of auxiliary basis functions

    Output:
        dense_vk : array of shape (nao,nao); contains the exchange matrix
    */

    const unsigned int i_one = 1;
    const unsigned int nao_naux = nao * naux;
    const double d_one = 1.0;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, pair_id; // Pair index and identity
    unsigned int iao, jao; // AO indices
    const char trans = 'T';
    // Array pointers for lapack
    double * out_ptr;
    double * cderi_ptr;
    double * int_ptr; 

#pragma omp for schedule(static) 

    for (ipair = 0; ipair < npair; ipair++){
        cderi_ptr = sparse_cderi + (ipair * naux); // This points to a vector of length naux
        pair_id = nonzero_pair[ipair]; 
        iao = tril_iao[pair_id];
        jao = tril_jao[pair_id];
        if (iao % nthreads == ithread){ // Prevent race condition by assigning each CDERI output idx to 1 thread
            // I COULD make vk lower-triangular, but that speedup wouldn't parallel-scale unless I reworked the output thread assignment 
            out_ptr = dense_vk + (iao * nao); // This points to a vector of length nao
            int_ptr = dense_int + (jao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dgemv_(&trans, &naux, &nao, &d_one, int_ptr, &naux, cderi_ptr, &i_one, &d_one, out_ptr, &i_one);
            /* ^ I may have this backwards but if dgemv_ interprets everything in col-major form, this is what I need to get a
            row-major output of the type I want. Likewise below! */
        }
        if (iao != jao){ if (jao % nthreads == ithread){ 
            out_ptr = dense_vk + (jao * nao); // This points to a vector of length nao
            int_ptr = dense_int + (iao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dgemv_(&trans, &naux, &nao, &d_one, int_ptr, &naux, cderi_ptr, &i_one, &d_one, out_ptr, &i_one);
        }}
    }
}
}

void SDFKmatR (double * sparse_cderi, double * dense_dm, double * dense_vk, double * large_int, double * small_int, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    /*
    Wrapper for the three steps of calculating the exchange matrix from the sparse CDERI array and a density matrix:
    first contraction, transpose, second contraction

    Input:
        sparse_cderi : array of shape (npair, naux); contains CDERIs
        dense_dm : array of shape (nao, nao); contains density matrix
        large_int : array of shape (nao, nao, naux); used to contain the large, dense intermediate created after the first step
        small_int : array of shape (nthreads, naux); used to contain vectors when transposing large_int
        nonzero_pair : array of shape (npair) ; contains addresses of nao pairs with nonzero overlap
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        npair : number of pairs with nonzero overlap ; <= nao * (nao+1) / 2 ; propto nao in thermodynamic limit
        nao : number of AOs
        naux : number of auxiliary basis functions

    Output:
        dense_vk : array of shape (nao, nao); contains the exchange matrix
    */

    SDFKmatR1 (sparse_cderi, dense_dm, large_int, nonzero_pair, tril_iao, tril_jao, npair, nao, naux);
    SDFKmatRT (large_int, small_int, tril_iao, tril_jao, nao, naux);
    SDFKmatR2 (sparse_cderi, large_int, dense_vk, nonzero_pair, tril_iao, tril_jao, npair, nao, naux);

}

void SINT_SDCDERI_DDMAT (double * dense_cderi, double * dense_A, double * dense_prod, double * cderi_wrk, double * A_wrk, 
    int * iao_sort, int * iao_nent, int * iao_entlist, int nao, int naux, int nmo, int nent_max)
{
    /*
    Matrix-multiply a CDERI array using sparsity information from the CDERIs, but not the other multiplicand.

    Input:
        dense_cderi : array of shape (naux, nao*(nao+1)/2); contains CDERIs (dense, lower-triangular storage with auxiliary basis index first)
        dense_A : array of shape (nao, nmo); contains the multiplicand
        iao_sort : array of shape (nao); sorts the AOs according to iao_nent (for benefit of parallel scaling)
        iao_nent : array of shape (nao); lists how many nonvanishing pairs exist for each orbital
        iao_entlist : array of shape (nao, nent_max); lists the other orbital in nonvanishing pairs involving each orbital

    Input/output:
        cderi_wrk : array of shape (nthreads, nent_max, naux); contains 0s on input and garbage on output
        A_wrk : array of shape (nthreads, nent_max, nmo); contains 0s on input and garbage on output
        
    Output:
        dense_prod : array of shape (nao, nmo, naux); contains result of multiplication (with auxbasis index placed last) 
        Transpose me BEFORE calling SINT_SDCDERI_VK!
    */

    
    const unsigned int i_one = 1;
    const unsigned int npair = nao * (nao + 1) / 2;
    const double d_one = 1.0;
    const char transCDERI = 'T';
    const char transA = 'N';

    
#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int iao_ix, iao, jao_ix, jao; // AO indices
    unsigned int my_nent;
    const char trans = 'T';
    // Array pointers for lapack
    double * my_cderi; // put nent on the faster-moving index: (naux, nent_max)
    double * my_A; // put nent on the faster-moving index: (nmo, nent_max) in C
    double * my_prod;
    double * my_cderi_wrk;
    double * my_A_wrk;
    int * my_entlist;
    my_cderi_wrk = cderi_wrk + (ithread * nent_max * naux);
    my_A_wrk = A_wrk + (ithread * nent_max * nmo);

#pragma omp for schedule(static) 

    for (iao_ix = 0; iao_ix < nao; iao_ix++){
        iao = iao_sort[iao_ix];
        my_nent = iao_nent[iao];
        my_entlist = iao_entlist + (iao*nent_max);
        my_prod = dense_prod + (iao * nmo * naux);
        for (jao_ix = 0; jao_ix < my_nent; jao_ix++){
            jao = my_entlist[jao_ix];
            if (iao > jao){ 
                my_cderi = dense_cderi + ((iao * (iao + 1) / 2) + jao);
            } else {
                my_cderi = dense_cderi + ((jao * (jao + 1) / 2) + iao);
            }
            my_A = dense_A + (jao * nmo);
            dcopy_(&naux, my_cderi, &npair, my_cderi_wrk + jao_ix, &my_nent);
            dcopy_(&nmo, my_A, &i_one, my_A_wrk + jao_ix, &my_nent);
        }
        dgemm_(&transCDERI, &transA, &naux, &nmo, &my_nent,
            &d_one, my_cderi_wrk, &my_nent, my_A_wrk, &my_nent,
            &d_one, my_prod, &naux);
        /* Remember, dgemm_ is Fortran and reads array indices backwards. So when I say
        I am transposing my_CDERI but not transposing my_A, it's actually the opposite. */
    }

}
}

void SINT_SDCDERI_VK (double * dense_cderi, double * dense_int, double * dense_vk, double * cderi_wrk, double * int_wrk, 
    int * iao_sort, int * iao_nent, int * iao_entlist, int nao, int naux, int nent_max)
{

    /* SDFKmatR2 is much faster than this, I guess because of the dcopy_'s
    Compute the exchange matrix: k(m,n) = sum(p,r,s) CDERI(p,m,r) [CDERI(p,n,s) DM(r,s)]

    Input:
        dense_cderi : array of shape (nao*(nao+1)/2, naux); contains CDERIs (dense, lower-triangular storage with auxiliary basis index fast)
        dense_int : array of shape (nao, nao, naux); contains the intermediate I(r,n,p) = sum(s) CDERI(p,n,s) DM(r,s)
        iao_sort : array of shape (nao); sorts the AOs according to iao_nent (for benefit of parallel scaling)
        iao_nent : array of shape (nao); lists how many nonvanishing pairs exist for each orbital
        iao_entlist : array of shape (nao, nent_max); lists the other orbital in nonvanishing pairs involving each orbital

    Input/output:
        cderi_wrk : array of shape (nthreads, nent_max, naux); contains 0s on input and garbage on output
        int_wrk : array of shape (nthreads, nent_max, naux, nao); contains 0s on input and garbage on output
            This may be a serious memory-consumption problem!
        
    Output:
        dense_vk : array of shape (nao, nao); contains exchange matrix 
    */

    const unsigned int i_one = 1;
    const unsigned int npair = nao * (nao + 1) / 2;
    const unsigned int nao_naux = nao * naux;
    const double d_one = 1.0;
    const char transINT = 'T';

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int iao_ix, iao, jao_ix, jao; // AO indices
    unsigned int my_nent;
    const char trans = 'T';
    // Array pointers for lapack
    double * my_cderi; 
    double * my_int; 
    double * my_vk;
    double * my_cderi_wrk;
    double * my_int_wrk;
    int * my_entlist;
    int nent_naux;
    my_cderi_wrk = cderi_wrk + (ithread * nent_max * naux);
    my_int_wrk = int_wrk + (ithread * nent_max * naux * nao);

#pragma omp for schedule(static) 

    for (iao_ix = 0; iao_ix < nao; iao_ix++){
        iao = iao_sort[iao_ix];
        my_nent = iao_nent[iao];
        my_entlist = iao_entlist + (iao*nent_max);
        my_vk = dense_vk + (iao * nao);
        nent_naux = my_nent * naux;
        for (jao_ix = 0; jao_ix < my_nent; jao_ix++){
            jao = my_entlist[jao_ix];
            if (iao > jao){ 
                my_cderi = dense_cderi + ((iao * (iao + 1) / 2) + jao) * naux;
            } else {
                my_cderi = dense_cderi + ((jao * (jao + 1) / 2) + iao) * naux;
            }
            my_int = dense_int + (jao * nao * naux);
            jao = iao + 1; // Fill lower-triangular part only 
            dgemv_(&transINT, &naux, &jao, &d_one, my_int, &naux, my_cderi, &i_one, &d_one, my_vk, &i_one);
            //dcopy_(&naux, my_cderi, &npair, my_cderi_wrk + jao_ix, &my_nent); 
            //dcopy_(&nao_naux, my_int, &i_one, my_int_wrk + jao_ix, &my_nent);   
        }
        //dgemv_(&transINT, &nent_naux, &nao, &d_one, my_int_wrk, &nent_naux, my_cderi_wrk, &i_one, &d_one, my_vk, &i_one);
    }
}
}






