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
from scipy import linalg, optimize
from pyscf import lib
from pyscf.fci import direct_spin1
from itertools import product
from mrh.exploratory.citools import fockspace
from mrh.my_pyscf.mcscf.lasci import all_nonredundant_idx
from itertools import product

verbose_lbjfgs = [-1,-1,-1,0,50,99,100,101,101,101]

def kernel (fci, h1, h2, norb, nelec, nlas=None, ci0_f=None,
            tol=1e-10, gtol=1e-4, max_cycle=15000, 
            orbsym=None, wfnsym=None, ecore=0, **kwargs):

    if nlas is None: nlas = getattr (fci, 'nlas', norb)
    verbose = kwargs.get ('verbose', getattr (las, 'verbose', 0))
    las = fci.get_obj (fci, h1, h2, ci0_f, norb, nlas, ecore=ecore)
    las_options = {'ftol':     tol,
                   'gtol':     gtol,
                   'maxfun':   max_cycle,
                   'maxiter':  max_cycle,
                   'iprint':   verbose_lbjfgs[verbose],
                   'callback': las.callback}
    res = optimize.minimize (las.e_tot, las.get_init_guess (),
        method='L-BFGS-B', jac=las.jac, options=options)
    e_tot = las.e_tot (res.x)
    ci1 = las.get_fcivec (res.x)
    return e_tot, ci1

def make_rdm12 (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((norb,norb))
    dm2 = np.zeros ((norb,norb,norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d1, d2 = direct_spin1.make_rdm12 (fci, ci, norb, nelec, **kwargs)
        dm1 += d1
        dm2 += d2
    return dm1, dm2


def LASCI_ObjectiveFunction (object):
    ''' Evaluate the energy and Jacobian of a LASSCF trial function parameterized in terms
        of unitary CC singles amplitudes and CI transfer operators. '''

    def __init__(self, fcisolver, h1, h2, ci0_f, norb, nlas, ecore=0):
        self.fcisolver = fcisolver
        self.ecore = ecore
        self.h = (ecore, h1, h2)
        self.ci0_f = [ci.copy () for ci in ci0_f]
        self.norb = norb
        self.nlas = nlas
        self.nfrags = len (nlas)
        assert (sum (nlas) == norb)
        self.uniq_orb_idx = all_nonredundant_idx (norb, 0, nlas)
        offs = np.cumsum (nlas)
        offs = np.stack ([offs-np.array (nlas), offs], axis=1)

    def fermion_spin_shuffle (self, c, norb=None, nlas=None):
        if norb is None: norb = self.norb
        if nlas is None: nlas = self.nlas
        return c * fockspace.fermion_spin_shuffle (norb, nlas)

    def fermion_frag_shuffle (self, c, i, j, norb=None):
        if norb is None: norb=self.norb
        return c * fockspace.fermion_frag_shuffle (norb, i, j)

    def pack (self, xorb, xci_f):
        x = [xorb[self.uniq_orb_idx],]
        ci0_f = self.ci0_f
        for xci, ci0 in zip (xci_f, ci0_f):
            cHx = ci0.conj ().ravel ().dot (xci.ravel ())
            x.append ((xci - (ci0*cHx)).ravel ())
        return np.concatenate (x)

    def unpack (self, x):
        xorb = np.zeros ((self.norb, self.norb), dtype=x.dtype)
        xorb[self.uniq_orb_idx] = x[:self.nvar_orb]
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

    @property
    def nvar_tot (self):
        return self.nvar_orb + sum ([c.size for c in self.ci0_f])

    def energy_tot (self, x):
        uc, huc = self.hc_x (x)[:2]
        cu = uc.conj ().ravel ()
        hc = hc.ravel ()
        cuuc = lib.dot (cu, uc)
        cuhuc = lib.dot (cu, huc)
        return cuhuc/cuuc

    def jac (self, x):
        uc, huc, ints = self.hc_x (x)
        h, uc_f = ints[0]
        jact1 = self.get_jac_t1 (h, uc)
        jacci_f = self.get_jac_ci (uc_f, huc)
        return self.pack (jact1, jacci_f)

    def hc_x (self, x):
        xorb, xci = self.unpack (x)
        h = self.rotate_orb (self.h, xorb)
        uc_f = self.rotate_ci0 (xci)
        uc = self.dp_ci (uc_f)
        huc = self.contract_h2 (h, uc)
        return uc, huc, [h, uc_f]
        
    def contract_h2 (self, h, ci):
        norb = self.norb
        hci = h[0] * ci
        for neleca, nelecb in product (range (norb+1), repeat=2):
            nelec = (neleca, nelecb)
            h2eff = self.fcisolver.absorb_h1e (h[1], h[2], norb, nelec, 0.5)
            ci_h = fockspace.fock2hilbert (ci, norb, nelec)
            hc = direct_spin1.contract_2e (h2eff, ci_h, norb, nelec)
            hci += fockspace.hilbert2fock (hc, norb, nelec)
        return hci
            
    def dp_ci (self, ci_f):
        norb, nlas = self.norb, self.nlas
        ci = np.ones ([1,1], dtype=ci_f[0].dtype)
        for ix, c in enumerate(ci_f):
            ndet = 2**sum(nlas[:ix+1])
            ci = np.multiply.outer (c, ci).transpose (0,2,1,3).reshape (ndet, ndet)
        ci = self.fermion_spin_shuffle (ci, norb=norb, nlas=nlas)
        return ci

    def rotate_orb (self, h, xorb):
        umat = linalg.expm (xorb/2)
        h = [h[0], h[1].copy (), h[2].copy ()]
        h[1] = umat.conj ().T @ h[1] @ umat
        h[2] = np.tensordot (h[2], umat.conj (), axes=((0),(0))) # pqrs -> qrsi
        h[2] = np.tensordot (h[2], umat,         axes=((0),(0))) # qrsi -> rsij
        h[2] = np.tensordot (h[2], umat.conj (), axes=((0),(0))) # rsij -> sijk
        h[2] = np.tensordot (h[2], umat,         axes=((0),(0))) # sijk -> ijkl
        return h

    def rotate_ci0 (self, xci_f):
        ci0, norb = self.ci0_f, self.norb
        ci1 = []
        for dc, c in zip (xci, ci0):
            phi = linalg.norm (x)
            cosp = np.cos (phi)
            if np.abs (phi) > 1e-8: sinp = np.sin (phi) / phi
            else: sinp = 1 # as precise as it can be w/ 64 bits
            ci1.append (cosp*c + sinp*dc)
        if t1 is not None:
        return ci1

    def rotate_ci_t1 (self, ci1, t1):
        umat = linalg.expm (t1/2) 
        ci2 = np.zeros_like (ci1)
        for neleca, nelecb in product (range (norb+1), repeat=2):
            nelec = (neleca, nelecb)
            c = fockspace.fock2hilbert (ci1, norb, nelec)
            c = self.fci.transform_ci_for_orbital_rotation (c, norb, nelec, umat)
            ci2 += fockspace.hilbert2fock (ci1, norb, nelec)
        return ci2

    def get_jac_t1 (self, h, ci):
        norb = self.norb
        dm1, dm2 = self.fcisolver.make_rdm12 (ci, norb, 0)
        h1 = h[1]
        h2 = h[2].reshape (norb, norb**3)
        dm2 = dm2.reshape (norb, norb**3)
        f1 = lib.dot (h1, dm1.T) + lib.dot (h2, dm2.T)
        return f1 - f1.T

    def get_jac_ci (self, uci_f, huci):
        # "uci": U|ci0>
        # "huci": e^-T1 H e^T1 U|ci0>
        # "jacci": Jacobian elements for the CI degrees of freedom
        # subscript_f means a list over fragments
        # subscript_i means this is not a list but it applies to a particular fragment
        norb, nlas, ci0_f = self.norb, self.nlas, self.ci0_f
        jacci_f = []
        for ifrag, ci0 in enumerate (ci0_f):
            huci_i = huci.copy ()
            # We're going to contract the fragments in ascending order
            # so minor axes disappear first
            # However, we have to skip the ith fragment itself
            norb0, ndet0 = norb, 2**norb
            norb2, ndet2 = 0, 1
            for jfrag, (uci, norb1) in enumerate (uci_f, nlas):
                norb0 -= norb1
                if (jfrag==ifrag):
                    norb2 = norb1
                    ndet2 = 2**norb1
                    continue
                # norb0, norb1, and norb2 are the number of orbitals in the sectors arranged
                # in major-axis order: the slower-moving orbital indices we haven't touched yet,
                # the orbitals we are integrating over in this particular cycle of the for loop,
                # and the fast-moving orbital indices that we have to keep uncontracted
                # because they correspond to the outer for loop.
                # We want to move the field operators corresponding to the generation of the
                # norb1 set to the front of the operator products in order to integrate
                # with the correct sign.
                huci_i = self.fermion_frag_shuffle (huci_i, norb2, norb2+norb1, norb=norb0+norb1+norb2)
                ndet0 = 2**norb0
                ndet1 = 2**norb1
                huci_i = huci_i.reshape (ndet0, ndet1, ndet2, ndet0, ndet1, ndet2)
                huci_i = np.tensordot (huci_i, uci, axes=((1,4),(0,1))).reshape (ndet0*ndet2, ndet0*ndet2)
            # Remove parallel component!
            cH = ci0.conj ().ravel ()
            chuc = cH.dot (huci_i.ravel ())
            huci_i -= ci0 * chuc
            jacci_f.append (hci)
        return jacci_f

    def callback (self, x):
        pass

    def get_init_guess (self):
        return np.zeros (self.nvar_tot)

    def get_fcivec (self, x):
        xorb, xci_f = self.unpack (x)
        uc_f = self.rotate_ci0 (xci_f)
        uc = self.dp_ci (uc_f)
        uc = self.rotate_ci_t1 (uc, -xorb) # I hope the negative sign accomplishes the transpose...
        return uc / linalg.norm (uc)

class FCISolver (direct_spin1.FCISolver):
    kernel = kernel
    approx_kernel = kernel
    make_rdm12 = make_rdm12
    get_obj = LASCI_ObjectiveFunction
