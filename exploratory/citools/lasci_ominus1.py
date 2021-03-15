# Essentially a deoptimized implementation of LASSCF which is just a
# FCISolver object that does everything in the Fock-space FCI basis
# with no constraint for spin or charge.
#
# Must define:
#   kernel = approx_kernel
#   make_rdm12
#   get_init_guess
#
# In these functions, except for get_init_guess, "nelec" is ignored

import numpy as np
from scipy import linalg
from pyscf.fci import direct_spin1
from itertools import product
from mrh.exploratory.citools import fockspace
from mrh.my_pyscf.mcscf.lasci import all_nonredundant_idx

def kernel (fci, h1, h2, norb, nelec, ci0=None,
            tol=None, lindep=None, max_cycle=None, max_space=None,
            nroots=None, davidson_only=None, pspace_size=None,
            orbsym=None, wfnsym=None, ecore=0, **kwargs):
    pass

def make_rdm12 (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((norb,norb))
    dm2 = np.zeros ((norb,norb,norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d1, d2 = direct_spin1.make_rdm12 (fci, ci, norb, nelec, **kwargs)
        dm1 += d1
        dm2 += d2
    return dm1, dm2

class FCISolver (direct_spin1.FCISolver):
    kernel = kernel
    approx_kernel = kernel
    make_rdm12 = make_rdm12

class LASCI_UnitaryGroupGenerators (object):
    ''' Object for packing (for root-finding algorithms) and unpacking (for direct manipulation)
    the nonredundant variables ('unitary group generators') of a LASCI problem. Selects nonredundant
    lower-triangular part ('x') of a skew-symmetric orbital rotation matrix ('kappa') and projects
    away components of the ci vector parallel to the ground state. '''

    def __init__(self, ci0_f, norb, nlas):
        self.ci0_f = [c.copy () for c in ci0_f]
        self.norb = norb
        self.nlas = nlas
        self.dtype = ci0_f[0].dtype
        assert (sum (nlas) == norb)
        self.uniq_orb_idx = all_nonredundant_idx (norb, 0, nlas)

    def pack (self, xorb, xci_f):
        x = xorb[self.uniq_orb_idx]
        for x1, c1 in zip (xci_f, self.ci0_f):
            s = x1.ravel ().dot (c1.conj ().ravel ())
            x1 -= c1 * s
            x = np.append (x, x1.ravel ())
        return x

    def unpack (self, x):
        xorb = np.zeros ((self.norb, self.norb), dtype=x.dtype)
        sorb[self.uniq_orb_idx] = x[:self.nvar_orb]
        xorb = xorb - xorb.T

        y = x[self.nvar_orb:]
        xci = []
        for n in self.nlas:
            xci.append (y[:2**(2*n)].reshape (2**n, 2**n))
            y = y[2**(2*n):]

        return xorb, xci

    @property
    def nvar_orb (self):
        return np.count_nonzero (self.uniq_orb_idx)

    


