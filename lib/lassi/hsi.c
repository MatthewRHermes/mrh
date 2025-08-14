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

void HSIxyzmn_d (double * C, double * A, double * B,
                 const size_t X, const size_t Y, const size_t Z,
                 const size_t M, const size_t N, const size_t K)
{
/* Perform the einsum operation
   YMK,XZNK->XYZMN
   in a multithreaded fashion without any transposes.
   Input:
        A : array of shape (Y,M,K)
        B : array of shape (X,Z,N,K)
   Output:
        C : array of shape (X,Y,Z,M,N)
*/
const size_t major_dim = X * Y * Z;
const double d_one = 1.0;
#pragma omp parallel
{
    size_t iX,iY,iZ;
    double * myA;
    double * myB;
    double * myC;
    #pragma omp for schedule(static)
    for (size_t imajor = 0; imajor < major_dim; ++imajor){
        myC = C + imajor;
        iz = imajor % Z;
        iy = imajor / Z;
        ix = iy / Y;
        iy = iy % Y;
        myA = A + iY;
        myB = B + iX*iZ;
        // I want the minor index to be from B, which (since dgemm_ thinks like Fortran)
        // means I need to swap B and A. The axis being contracted is the minor axis. 
        dgemm_('T', 'N', &N, &M, &K, &d_one, myB, &K, myA, &K, &d_one, myC, &N);
    }
}
}

