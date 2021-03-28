import numpy as np
from scipy import linalg
from itertools import product
from mrh.exploratory.citools import fockspace


class LASUCCEffectiveHamiltonian (object):
    def __init__(self, full):
        self.ndet = ndet = full.shape[0]
        self.norb = int (round (np.log2 (ndet)))
        self.full = full.reshape (ndet*ndet, ndet*ndet)

    def get_nonzero_elements (self):
        idx = np.abs (self.full) > 1e-8
        nonzero = self.full[idx]
        return nonzero, idx

    def get_number_block (self, ni, nj):
        ndet, full = self.ndet, self.full
        ADDRS_NELEC = fockspace.ADDRS_NELEC
        if isinstance (ni, (int, np.integer)):
            ix = ADDRS_NELEC[:ndet**2] == ni
        else:
            ixa = ADDRS_NELEC[:ndet] == ni[0]
            ixb = ADDRS_NELEC[:ndet] == ni[1]
            ix = np.logical_and.outer (ixa, ixb).ravel ()
        if isinstance (nj, (int, np.integer)):
            jx = ADDRS_NELEC[:ndet**2] == nj
        else:
            jxa = ADDRS_NELEC[:ndet] == nj[0]
            jxb = ADDRS_NELEC[:ndet] == nj[1]
            jx = np.logical_and.outer (jxa, jxb).ravel ()
        idx = np.ix_(ix,jx)
        block = self.full[idx]
        return block, idx

def gen_frag_basis (psi, ifrag, ci_f=None):
    if ci_f is None: ci_f = psi.ci0_f
    norb, nlas, nfrag = psi.norb, psi.nlas, psi.nfrag
    ci0 = np.ones ([1,1], dtype=ci_f[0].dtype)
    n = 0
    for jfrag, (c, dn) in enumerate (zip (ci_f, nlas)):
        if (ifrag == jfrag): continue
        n += dn
        ndet = 2**n
        ci0 = np.multiply.outer (c, ci0).transpose (0,2,1,3).reshape (ndet, ndet)
    n = nlas[ifrag]
    ndet = 2**norb
    ci1 = np.empty ((ndet, ndet), dtype=ci_f[0].dtype)
    norb0 = sum (nlas[ifrag+1:]) if ifrag+1<nfrag else 0 
    norb1 = nlas[ifrag]
    norb2 = sum (nlas[:ifrag]) if ifrag>0 else 0
    ndet0, ndet1, ndet2 = 2**norb0, 2**norb1, 2**norb2
    for ideta, idetb in product (range (ndet1), repeat=2):
        ci1[:,:] = 0
        ci1 = ci1.reshape (ndet0, ndet1, ndet2, ndet0, ndet1, ndet2)
        ci1[:,ideta,:,:,idetb,:] = ci0.reshape (ndet0, ndet2, ndet0, ndet2)
        ci1 = ci1.reshape (ndet, ndet)
        ci1 = psi.fermion_spin_shuffle (ci1)
        yield ideta, idetb, ci1

def get_dense_heff (psi, x, h, ifrag):
    uop, nlas = psi.uop, psi.nlas
    xconstr, xcc, xci = psi.unpack (x)
    uop.set_uniq_amps_(xcc)
    ci_f = psi.rotate_ci0 (xci)
    h = psi.constr_h (xconstr, h)
    ndet = 2**nlas[ifrag]
    heff = np.zeros ((ndet,ndet,ndet,ndet), dtype=x.dtype)
    for ideta, idetb, c in psi.gen_frag_basis (ifrag, ci_f=ci_f):
        uhuc = uop (c)
        uhuc = psi.contract_h2 (h, uhuc)
        uhuc = uop (uhuc, transpose=True)
        heff[ideta,idetb,:,:] = psi.project_frag (ifrag, uhuc, ci0_f=ci_f)
    return LASUCCEffectiveHamiltonian (heff)

        


