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

void get_strides (long * arr_strides, int * arr_shape, int ndim)
{
    arr_strides[ndim-1] = 1;
    for (int i=ndim-2; i >= 0; i--){
        arr_strides[i] = arr_strides[i+1] * ((long) arr_shape[i+1]);
    }
}

void LASSIRDMddotinterleave (double * c, double * a, double * b,
                             long * c_strides, long * a_strides, long * b_strides,
                             int ndim, int c_nrows, int m, int n, int k)
{
/* evaluate, i.e., c=np.einsum ('ab...cdx,ij...klx->aibj...ckdl',a,b)

   Input:
        a : array of shape a_shape
        b : array of shape b_shape
        c_strides: array of shape (ndim)
            Offsets for each increment of each dimension of c in decreasing order
        a_strides: array of shape (ndim)
            Offsets for each increment of each dimension of a
        b_strides: array of shape (ndim)
            Offsets for each increment of each dimension of b
        ndim : integer
            Number of dimensions of a and b
        c_nrows : integer
            Length of major dimension of c
        m : integer
            Length of second-to-last dimension of a and c
        n : integer
            Length of last dimension of c and second-to-last dimension of b
        k : integer
            Length of last dimension of a and b

   Output:
       c : array shaped according to c_strides
*/
const double d_one = 1.0;
const int i_one = 1;
const char tran = 'T';
const char notr = 'N';
const long c_size = c_strides[0] * ((long) c_nrows);
const long c_ncols = ndim > 2 ? c_strides[2*(ndim-1)-3] : c_size;
#pragma omp parallel
{
    double * a_ptr;
    double * b_ptr;
    double * c_ptr;
    long idx, a_idx, b_idx;
    #pragma omp for
    for (long c_irow = 0; c_irow < c_size; c_irow += c_ncols){
        c_ptr = c + c_irow;
        a_ptr = a;
        b_ptr = b;
        idx = c_irow;
        for (int idim = 0; idim < ndim-2; idim++){
            a_idx = idx / c_strides[2*idim];
            a_ptr += a_strides[idim] * a_idx;
            idx = idx % c_strides[2*idim];
            b_idx = idx / c_strides[(2*idim)+1];
            b_ptr += b_strides[idim] * b_idx;
            idx = idx % c_strides[(2*idim)+1];
        }
        dgemm_(&tran, &notr, &n, &m, &k, &d_one, b_ptr, &k, a_ptr, &k, &d_one, c_ptr, &n);
    }
}
}

void LASSIRDMzdotinterleave (double complex * c, double complex * a, double complex * b,
                             int ndim, int * a_shape, int * b_shape)
{
/* evaluate, i.e., c=np.einsum ('ab...cdx,ij...klx->aibj...ckdl',a,b)

   Input:
        a : array of shape a_shape
        b : array of shape b_shape
        ndim : integer
        a_shape : array of shape (ndim)
        b_shape : array of shape (ndim)

   Output:
       c : array of shape given by a_shape and b_shape interleaved, omitting the last dimension
*/
const double complex z_one = 1.0;
const int i_one = 1;
const char tran = 'T';
const char notr = 'N';
#pragma omp parallel
{
    double complex * a_ptr;
    double complex * b_ptr;
    double complex * c_ptr;
    int * c_shape = malloc (2*(ndim-1)*sizeof(int));
    for (int i = 0; i < ndim-1; i++){
        c_shape[2*i]     = a_shape[i];
        c_shape[(2*i)+1] = b_shape[i];
    }
    long * a_strides = malloc (ndim*sizeof(long));
    long * b_strides = malloc (ndim*sizeof(long));
    long * c_strides = malloc (2*(ndim-1)*sizeof(long));
    get_strides (a_strides, a_shape, ndim);
    get_strides (b_strides, b_shape, ndim);
    get_strides (c_strides, c_shape, 2*(ndim-1));

    long c_ncols = c_strides[2*(ndim-1)-3];
    long c_size = c_strides[0] * c_shape[0];
    long idx, a_idx, b_idx;
    #pragma omp for
    for (long c_irow = 0; c_irow < c_size; c_irow += c_ncols){
        c_ptr = c + c_irow;
        a_ptr = a;
        b_ptr = b;
        idx = c_irow;
        for (int idim = 0; idim < ndim-2; idim++){
            a_idx = idx / c_strides[2*idim];
            a_ptr += a_strides[idim] * a_idx;
            idx = idx % c_strides[2*idim];
            b_idx = idx / c_strides[(2*idim)+1];
            b_ptr += b_strides[idim] * b_idx;
            idx = idx % c_strides[(2*idim)+1];
        }
        zgemm_(&tran, &notr, &(b_shape[ndim-2]), &(a_shape[ndim-2]), &(a_shape[ndim-1]),
               &z_one, b_ptr, &(b_shape[ndim-1]), a_ptr, &(a_shape[ndim-1]),
               &z_one, c_ptr, &(b_shape[ndim-2]));
    }
    free (c_shape);
    free (a_strides);
    free (b_strides);
    free (c_strides);
}
}

void LASSIRDMdgetwgtfac (double * fac, double * wgt, double * sivec,
                         int nbas, int nroots,
                         long * bra, long * ket, int nelem)
{
/* Evaluate fac = np.dot (si[bra,:]*si[ket,:], wgt) with multiple threads

   Input:
        wgt : array of shape (nelem); contains overlap & permutation factors
        sivec : array of shape (nroots,nbas); contains SI vector coefficients
        bra : array of shape (nelem); identifies SI rows
        ket : array of shape (nelem); identifies SI rows

   Output:
        fac : array of shape (nroots)
*/
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
                     int nroots, int nelem_dest)
{
/* Evaluate SDdest = np.multiply.outer (fac, SDsrc) with multiple threads.

   Input:
        fac : array of shape (nroots)
        SDsrc : array of shape (nelem_dest)

   Output:
        SDdest : array of shape (nroots,nelem_dest)
*/
const unsigned int i_one = 1;
#pragma omp parallel
{
    int nt = omp_get_num_threads ();
    int it = omp_get_thread_num ();
    int nel0 = nelem_dest / nt;
    int nel1 = nelem_dest % nt;
    int tstart = (nel0*it) + MIN(nel1,it);
    int tlen = it<nel1 ? nel0+1 : nel0;
    double * mySDsrc = SDsrc+tstart;
    double * mySDdest = SDdest+tstart;
    for (int iroot = 0; iroot < nroots; iroot++){
        daxpy_(&(tlen), fac+iroot,
               mySDsrc, &i_one,
               mySDdest+iroot*nelem_dest, &i_one);
    }
}
}

void LASSIRDMdputSD1 (double * SDdest, double * SDsrc,
                      int nroots, int ndest, int nsrc,
                      int * SDdest_idx, int * SDsrc_idx, int * SDlen,
                      int nidx)
{
/* Add to segmented elements of RDM1s arrays from a contiguous source array

   Input:
        SDsrc : array of shape (nroots,2,nsrc*nsrc)
        SDdest_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDdest
        SDsrc_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDsrc
        SDlen : array of shape (nidx); lengths of contiguous blocks

   Input/Output:
        SDdest : array of shape (nroots,2,ndest*ndest)
            Elements of SDsrc corresponding to SDdest_idx are added
*/
const unsigned int i_one = 1;
const double d_one = 1.0;
const int npdest = ndest*ndest;
const int npsrc = nsrc*nsrc;
const int nblk = nroots*2*nidx;
const int lroot = 2*nidx;
const int SDsrc_size = nroots*2*npsrc;
const int SDdest_size = nroots*2*npdest;
#pragma omp parallel
{
    int iroot, ispin, iidx, j;
    int sidx, didx;
    #pragma omp for 
    for (int i = 0; i < nblk; i++){
        iroot = i/lroot;
        j = i%lroot;
        ispin = j/nidx;
        iidx = j%nidx;
        assert (iroot < nroots);
        assert (ispin < 2);
        assert (iidx < nidx);
        sidx = (iroot*2 + ispin)*npsrc + SDsrc_idx[iidx];
        didx = (iroot*2 + ispin)*npdest + SDdest_idx[iidx];
        assert (sidx < SDsrc_size);
        assert (didx < SDdest_size);
        daxpy_(SDlen+iidx, &d_one, SDsrc+sidx, &i_one, SDdest+didx, &i_one);
    }
}
}

void LASSIRDMdputSD2 (double * SDdest, double * SDsrc,
                      int nroots, int ndest, int nsrc, int * pdest,
                      int * SDdest_idx, int * SDsrc_idx, int * SDlen,
                      int nidx)
{
/* Add to segmented elements of RDM2s arrays from a contiguous source array

   Input:
        SDsrc : array of shape (nroots,4,nsrc*nsrc,nsrc*nsrc)
        pdest : array of shape (nsrc,nsrc)
            Indices of all addressed elements in the second-minor dimension of SDdest
        SDdest_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDdest
        SDsrc_idx : array of shape (nidx)
            beginnings of contiguous blocks in the minor dimension of SDsrc
        SDlen : array of shape (nidx); lengths of contiguous blocks

   Input/Output:
        SDdest : array of shape (nroots,2,ndest*ndest,ndest*ndest)
            Elements of SDsrc corresponding to SDdest_idx are added
*/
const unsigned int i_one = 1;
const double d_one = 1.0;
const int npdest = ndest*ndest;
const int npsrc = nsrc*nsrc;
const long ntdest = npdest*npdest;
const long ntsrc = npsrc*npsrc;
const int nblk = nroots*4*npsrc*nidx;
const int lroot = 4*npsrc*nidx;
const int lspin = npsrc*nidx;
const long SDsrc_size = nroots*4*ntsrc;
const long SDdest_size = nroots*4*ntdest;
#pragma omp parallel
{
    int ipdest, iroot, ispin, iidx, j;
    long sidx, didx;
    #pragma omp for 
    for (int i = 0; i < nblk; i++){
        iroot = i/lroot;
        j = i%lroot;
        ispin = j/lspin;
        j = j%lspin;
        ipdest = j/nidx;
        iidx = j%nidx;
        assert (iroot < nroots);
        assert (ispin < 4);
        assert (ipdest < npsrc);
        assert (iidx < nidx);
        sidx = ((long) (iroot*4 + ispin))*ntsrc + ((long) ipdest*npsrc + SDsrc_idx[iidx]);
        didx = ((long) (iroot*4 + ispin))*ntdest + ((long) pdest[ipdest]*npdest + SDdest_idx[iidx]);
        assert (sidx < SDsrc_size);
        assert (didx < SDdest_size);
        daxpy_(SDlen+iidx, &d_one, SDsrc+sidx, &i_one, SDdest+didx, &i_one);
    }
}
}
