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
const size_t XYZ = X*Y*Z;
const double d_one = 1.0;
const double d_zero = 0.0;
const size_t MN = M * N;
const size_t MK = M * K;
const size_t NK = N * K;
const size_t ZNK = Z * NK;
const char charT = 'T';
const char charN = 'N';
const int N1 = N;
const int M1 = M;
const int K1 = K;
#pragma omp parallel
{
    size_t iX,iY,iZ;
    double * myA;
    double * myB;
    double * myC;
    #pragma omp for schedule(static)
    for (size_t iXYZ = 0; iXYZ < XYZ; ++iXYZ){
        myC = C + (iXYZ*MN);
        iZ = iXYZ % Z;
        iY = iXYZ / Z; // iXY
        iX = iY / Y;
        iY = iY % Y;
        myA = A + (iY*MK);
        myB = B + (iX*ZNK) + (iZ*NK);
        // I want the minor index to be from B, which (since dgemm_ thinks like Fortran)
        // means I need to swap B and A. The axis being contracted is the minor axis. 
        dgemm_(&charT, &charN,
               &N1, &M1, &K1, &d_one,
               myB, &K1,
               myA, &K1,
               &d_zero, myC, &N1);
    }
}
}

void HSIxyzmnT_d (double * C, double * A, double * B,
                 const size_t X, const size_t Y, const size_t Z,
                 const size_t M, const size_t N, const size_t K)
{
/* Perform the einsum operation
   YMK,KZNX->XYZMN
   in a multithreaded fashion without any transposes.
   Input:
        A : array of shape (Y,M,K)
        B : array of shape (K,Z,N,X)
   Output:
        C : array of shape (X,Y,Z,M,N)
*/
const size_t XYZ = X*Y*Z;
const size_t XYZN = XYZ*N;
const double d_one = 1.0;
const double d_zero = 0.0;
const size_t MN = M * N;
const size_t MK = M * K;
const size_t NX = N * X;
const size_t ZNX = Z * NX;
const char charT = 'T';
const char charN = 'N';
const int N1 = N;
const int M1 = M;
const int K1 = K;
const int ZNX1 = ZNX; // wish upon a star that this doesn't roll over
#pragma omp parallel
{
    size_t iXYZ;
    size_t iX,iY,iZ,iN;
    double * myA;
    double * myB;
    double * myC;
    #pragma omp for schedule(static)
    for (size_t iXYZN = 0; iXYZN < XYZN; ++iXYZN){
        iXYZ = iXYZN / N;
        iN = iXYZN % N;
        iZ = iXYZ % Z;
        iY = iXYZ / Z; // iXY
        iX = iY / Y;
        iY = iY % Y;
        myA = A + (iY*MK);
        myB = B + (iZ*NX) + (iN*X) + iX;
        myC = C + (iXYZ*MN) + iN;
        // A -> A, which is thankfully contiguous
        // B -> x, with stride ZNX
        // C -> y, with stride N
        dgemv_(&charT, &K1, &M1, &d_one, myA, &K1,
               myB, &ZNX1,
               &d_zero, myC, &N1); 
    }
}
}
