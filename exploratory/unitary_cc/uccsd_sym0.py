import numpy as np
import time, ctypes, math
from scipy import linalg
from mrh.lib.helper import load_library

libfsucc = load_library ('libfsucc')

# "sym0" means no spin or number symmetry at all: the full fock space,
# with a single flat CI vector corresponding to determinant strings
# identical to CI vector index numbers in a 64-bit unsigned integer format

'''
    fn(vv.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       mo.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int (nmo),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       pnon0tab, pshls_slice, pao_loc)
'''

def _op (norb, a_idxs, i_idxs, tamps, psi, inplace=True, transpose=False, derivs=[]):
    ''' Evaluate U|Psi> = ... e^t2 e^t1 e^t0 |Psi>, where tn are single
        generators

        Args:
            a_idxs : list of len (ngen)
                ndarrays listing +cr,-an operators
            i_idxs : list of len (ngen)
                ndarrays listing +an,-cr operators
            tamps : ndarray of shape (ngen)
                amplitudes/angles
            psi : ndarray of len (2**norb)
                spinless fock-space CI array

        Kwargs:
            inplace : logical
                Setting to False creates a copy
            transpose : logical
                If true, generators are applied in reverse order with
                amplitudes * -1
            derivs : list or ndarray
                List of amplitudes by which to differentiate (by adding pi/2)
                This function returns a ~single~ value or derivative vector
                of order len (derivs)

        Returns:
            upsi : ndarray of len (2**norb)
                new spinless fock-space CI array
    '''
    upsi = psi.view () if inplace else psi.copy ()
    sgn = 1 - (2*int (transpose))
    my_tamps = tamps * sgn
    for ideriv in derivs: my_tamps[ideriv] += math.pi * 0.5
    ngen = len (a_idxs)
    gen_range = range (ngen-1, -1, -1) if transpose else range (ngen)
    for igen in gen_range:
        libfsucc.FSUCCcontract1 (a_idxs[igen].ctypes.data_as (ctypes.c_void_p),
            i_idxs[igen].ctypes.data_as (ctypes.c_void_p),
            ctypes.c_double (sgn*tamps[igen]),
            upsi.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_uint (norb),
            ctypes.c_uint (len (a_idxs[igen])),
            ctypes.c_uint (len (i_idxs[igen])))
    return upsi

class FSUCCOperator (object):

    def __init__(self, norb, a_idxs, i_idxs):
        self.norb = norb
        self.ngen = ngen = len (a_idxs)
        self.a_idxs = [np.ascontiguousarray (a, dtype=np.uint8) for a in a_idxs]
        self.i_idxs = [np.ascontiguousarray (i, dtype=np.uint8) for i in i_idxs]
        assert (len (self.i_idxs) == ngen)
        self.amps = np.zeros (ngen)
        self.assert_sanity (nodupes=True)

    def assert_sanity (self, nodupes=True):
        ''' check for nilpotent generators, too many cr/an ops, or orbital
            indices out of range. nodupes -> check for duplicates under
            permutation symmetry (expensive) '''
        norb, ngen = self.norb, self.ngen
        pq_sorted = []
        for a, i in zip (self.a_idxs, self.i_idxs):
            p = np.append (i, a)
            errstr = 'a,i={},{} invalid for norb={}'.format (a,i,norb)
            assert (np.amax (p) < norb), errstr
            a_maxcnt = np.amax (np.unique (a, return_counts=True)[1])
            errstr = 'a={} is nilpotent'.format (a)
            assert (a_maxcnt < 2), errstr
            i_maxcnt = np.amax (np.unique (i, return_counts=True)[1])
            errstr = 'i={} is nilpotent'.format (a)
            assert (i_maxcnt < 2), errstr
            # passing these three implies that there aren't too many ops
            a_sorted, i_sorted = np.sort (a), np.sort (i)
            errstr = 'undefined amplitude detected (i==a)'
            assert (not np.all (a_sorted==i_sorted)), errstr
            pq_sorted.append (tuple (sorted ([tuple (a_sorted), tuple (i_sorted)])))
        if nodupes:
            pq_sorted = set (pq_sorted)
            errstr = 'duplicate generators detected'
            assert (len (pq_sorted) == ngen), errstr

    def __call__(self, ket, transpose=False, inplace=False):
        return _op(self.norb, self.a_idxs, self.i_idxs, self.amps, ket,
            inplace=inplace, transpose=transpose)

if __name__ == '__main__':
    norb = 4
    def pbin (n):
        s = bin (n)[2:]
        m = norb - len (s)
        if m: s = ''.join (['0',]*m) + s
        return s
    psi = np.zeros (2**norb)
    psi[3] = 1.0
    a = [np.array ([2], dtype=np.uint8), np.array ([0,2], dtype=np.uint8), np.array ([2,0], dtype=np.uint8)]
    i = [np.array ([1], dtype=np.uint8), np.array ([3,1], dtype=np.uint8), np.array ([3,2], dtype=np.uint8)]
    tamps = [math.pi/4, 0.33, 0.5]
    upsi = _op (norb, a, i, tamps, psi, inplace=False)
    for ix in range (2**norb):
        print (pbin (ix), psi[ix], upsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))
    upsi = _op (norb, a, i, tamps, upsi, transpose=True)
    for ix in range (2**norb):
        print (pbin (ix), psi[ix], upsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))

    a, i = np.tril_indices (norb, k=-1)
    uop = FSUCCOperator (norb, a, i)
    uop.amps = (1 - 2*np.random.rand (uop.ngen))*math.pi
    upsi = uop (psi)
    for ix in range (2**norb):
        print (pbin (ix), psi[ix], upsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))
