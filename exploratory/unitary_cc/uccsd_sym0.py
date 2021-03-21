import numpy as np
import time, ctypes, math
from scipy import linalg
from mrh.lib.helper import load_library
from itertools import combinations

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

def _op1_(norb, aidx, iidx, amp, psi, transpose=False, deriv=0):
    ''' Evaluates U|Psi> = e^(amp * [a0'a1'...i1'i0' - h.c.])|Psi>

        Args:
            norb : integer
                number of orbitals in the fock space
            aidx : list of len (na)
                lists +cr,-an operators
            iidx : list of len (ni)
                lists +an,-cr operators
            amp : float
                amplitude for generator
            psi : ndarray of len (2**norb)
                spinless fock-space CI array; modified in-place

        Kwargs:
            transpose : logical
                Setting to True multiplies the amp by -1
            deriv: int
                Order of differentiation wrt the amp

        Returns:
            psi : ndarray of len (2**norb)
                arg "psi" after operation
    '''
    assert (psi.flags['C_CONTIGUOUS'])
    sgn = 1 - (2*int (transpose))
    my_amp = sgn * (amp + (deriv * math.pi / 2))
    aidx = np.ascontiguousarray (aidx, dtype=np.uint8)
    iidx = np.ascontiguousarray (iidx, dtype=np.uint8)
    na, ni = aidx.size, iidx.size
    aidx_ptr = aidx.ctypes.data_as (ctypes.c_void_p)
    iidx_ptr = iidx.ctypes.data_as (ctypes.c_void_p)
    psi_ptr = psi.ctypes.data_as (ctypes.c_void_p)
    libfsucc.FSUCCcontract1 (aidx_ptr, iidx_ptr,
        ctypes.c_double (my_amp), psi_ptr,
        ctypes.c_uint (norb),
        ctypes.c_uint (na),
        ctypes.c_uint (ni))
    return psi

def _projai_(norb, aidx, iidx, psi):
    ''' Project |Psi> into the space that interacts with the operators
        a1'a2'...i1i0 and i1'i2'...a1a0

        Args:
            norb : integer
                number of orbitals in the fock space
            aidx : list of len (na)
                lists +cr,-an operators
            iidx : list of len (ni)
                lists +an,-cr operators
            psi : ndarray of len (2**norb)
                spinless fock-space CI array; modified in-place

        Returns:
            psi : ndarray of len (2**norb)
                arg "psi" after operation
    '''
    aidx = np.ascontiguousarray (aidx, dtype=np.uint8)
    iidx = np.ascontiguousarray (iidx, dtype=np.uint8)
    na, ni = aidx.size, iidx.size
    aidx_ptr = aidx.ctypes.data_as (ctypes.c_void_p)
    iidx_ptr = iidx.ctypes.data_as (ctypes.c_void_p)
    psi_ptr = psi.ctypes.data_as (ctypes.c_void_p)
    libfsucc.FSUCCprojai (aidx_ptr, iidx_ptr, psi_ptr,
        ctypes.c_uint (norb),
        ctypes.c_uint (na),
        ctypes.c_uint (ni))
    return psi

class FSUCCOperator (object):

    def __init__(self, norb, a_idxs, i_idxs):
        self.norb = norb
        self.ngen = ngen = len (a_idxs)
        self.a_idxs = [np.ascontiguousarray (a, dtype=np.uint8) for a in a_idxs]
        self.i_idxs = [np.ascontiguousarray (i, dtype=np.uint8) for i in i_idxs]
        assert (len (self.i_idxs) == ngen)
        self.amps = np.zeros (ngen)
        self.assert_sanity (nodupes=True)

    def gen_fac (self, reverse=False):
        ''' Iterate over unitary factors/generators. '''
        ngen = self.ngen
        intr = int (reverse)
        start = 0 + (intr * (ngen-1))
        stop = ngen - (intr * (ngen+1))
        step = 1 - (2*intr)
        for igen in range (start, stop, step):
            yield igen, self.a_idxs[igen], self.i_idxs[igen], self.amps[igen]

    def gen_deriv1 (self, psi, transpose=False):
        ''' Iterate over first derivatives of U|Psi> wrt to generator amplitudes '''
        for igend in range (self.ngen):
            dupsi = psi.copy ()
            for ix, aidx, iidx, amp in self.gen_fac (reverse=transpose):
                if ix==igend: _projai_(self.norb, aidx, iidx, dupsi)
                _op1_(self.norb, aidx, iidx, amp, dupsi,
                    transpose=transpose, deriv=(ix==igend))
            yield dupsi

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
            a_maxcnt = np.amax (np.unique (a, return_counts=True)[1]) if len (a) else 0
            errstr = 'a={} is nilpotent'.format (a)
            assert (a_maxcnt < 2), errstr
            i_maxcnt = np.amax (np.unique (i, return_counts=True)[1]) if len (i) else 0
            errstr = 'i={} is nilpotent'.format (a)
            assert (i_maxcnt < 2), errstr
            # passing these three implies that there aren't too many ops
            a_sorted, i_sorted = np.sort (a), np.sort (i)
            errstr = 'undefined amplitude detected (i==a) {},{}'.format (i, a)
            if (len (a) and len (i)): assert (not np.all (a_sorted==i_sorted)), errstr
            pq_sorted.append (tuple (sorted ([tuple (a_sorted), tuple (i_sorted)])))
        if nodupes:
            pq_sorted = set (pq_sorted)
            errstr = 'duplicate generators detected'
            assert (len (pq_sorted) == ngen), errstr

    def __call__(self, psi, transpose=False, inplace=False):
        upsi = psi.view () if inplace else psi.copy ()
        for ix, aidx, iidx, amp in self.gen_fac (reverse=transpose):
            _op1_(self.norb, aidx, iidx, amp, upsi, transpose=transpose, deriv=0)
        return upsi

def get_uccs_op (norb, tp=None, tph=None):
    # This is incomplete
    p = list (range (norb))
    t1_idx = np.tril_indices (norb, k=-1)
    a, i = list (t1_idx[0]), list (t1_idx[1])
    a = p + a
    i = [[] for q in p] + i
    uop = FSUCCOperator (norb, a, i) 
    npair = norb * (norb - 1) // 2
    assert (len (t1_idx[0]) == npair)
    if tp is not None:
        uop.amps[:norb] = tp[:]
    if tph is not None:
        uop.amps[norb:][:npair] = tph[t1_idx] 
    return uop

def get_uccsd_op (norb, tp=None, tph=None, t2=None, uop_s=None):
    # This is incomplete
    if uop_s is None: uop_s = get_uccs_op (norb, tp=tp, tph=tph)
    init_offs = uop_s.ngen
    ab_idxs = uop_s.a_idxs
    ij_idxs = uop_s.i_idxs
    pq = [(p, q) for p, q in zip (*np.tril_indices (norb,k=-1))]
    a = []
    b = []
    i = []
    j = []
    for ab, ij in combinations (pq, 2):
        ab_idxs.append (ab)
        ij_idxs.append (ij)
        a.append (ab[0])
        b.append (ab[1])
        i.append (ij[0])
        j.append (ij[1])
    uop = FSUCCOperator (norb, ab_idxs, ij_idxs)
    uop.amps[:init_offs] = uop_s.amps[:]
    if t2 is not None:
        uop.amps[init_offs:] = t2[(a,i,b,j)]
    return uop

def get_uccs_op_numsym (norb, t1=None):
    a, i = np.tril_indices (norb, k=-1)
    uop = FSUCCOperator (norb, a, i)
    if t1 is not None: uop.amps[:] = t1[(a,i)]
    return uop

def get_uccsd_op_numsym (norb, t1=None, t2=None):
    uop_s = get_uccs_op_numsym (norb, t1=t1)
    return get_uccsd_op (norb, t2=t2, uop_s=uop_s)
    
if __name__ == '__main__':
    norb = 4
    def pbin (n):
        s = bin (n)[2:]
        m = norb - len (s)
        if m: s = ''.join (['0',]*m) + s
        return s
    psi = np.zeros (2**norb)
    psi[3] = 1.0

    #a, i = np.tril_indices (norb, k=-1)
    #uop = FSUCCOperator (norb, a, i)
    #uop.amps = (1 - 2*np.random.rand (uop.ngen))*math.pi
    tp_rand = np.random.rand (norb)
    tph_rand = np.random.rand (norb,norb)
    t2_rand = np.random.rand (norb,norb,norb,norb)
    uop_s = get_uccs_op (norb, tp=tp_rand, tph=tph_rand)
    upsi = uop_s (psi)
    uTupsi = uop_s (upsi, transpose=True)
    for ix in range (2**norb):
        print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))

    uop_sd = get_uccsd_op (norb, tp=tp_rand, tph=tph_rand, t2=t2_rand)
    upsi = uop_sd (psi)
    uTupsi = uop_sd (upsi, transpose=True)
    for ix in range (2**norb):
        print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))

    def obj_fun (x):
        uop_sd.amps[:] = x
        upsi = uop_sd (psi)
        err = upsi.dot (upsi) - (upsi[7]**2)
        jac = np.zeros_like (x)
        for ix, dupsi in enumerate (uop_sd.gen_deriv1 (psi)):
            jac[ix] += 2*(upsi.dot (dupsi) - dupsi[7]*upsi[7])
        return err, jac

    from scipy import optimize
    res = optimize.minimize (obj_fun, uop_sd.amps, method='BFGS', jac=True)

    print (res.success)
    uop_sd.amps[:] = res.x
    upsi = uop_sd (psi)
    uTupsi = uop_sd (upsi, transpose=True)
    for ix in range (2**norb):
        print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))

