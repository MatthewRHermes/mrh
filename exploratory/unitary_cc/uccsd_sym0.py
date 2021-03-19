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


if __name__ == '__main__':
    norb = 4 
    psi = np.zeros (2**norb)
    psi[3] = 1.0
    a = [np.array ([2], dtype=np.uint8), np.array ([0,2], dtype=np.uint8)]
    i = [np.array ([1], dtype=np.uint8), np.array ([3,1], dtype=np.uint8)]
    tamps = [math.pi/4, 0.33]
    upsi = _op (norb, a, i, tamps, psi, inplace=False)
    for i in range (2**norb):
        print (bin (i), psi[i], upsi[i])
    print (psi.dot (psi), upsi.dot (upsi))


