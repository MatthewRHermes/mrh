import numpy as np
import math, ctypes
from scipy import linalg
from mrh.lib.helper import load_library
from mrh.exploratory.unitary_cc import uccsd_sym0
from itertools import combinations, permutations, product, combinations_with_replacement

libfsucc = load_library ('libfsucc')

# Enforce n, sz, and first-order s**2 symmetry in unitary generators. First-
# order means t1_alpha = t2_beta. Unfortunately, I don't know how to ensure
# general s**2 symmetry

def spincases (p_idxs, norb):
    nelec = len (p_idxs)
    p_idxs = p_idxs[None,:]
    m = np.array ([0])
    for ielec in range (nelec):
        q_idxs = p_idxs.copy ()
        q_idxs[:,ielec] += norb
        p_idxs = np.append (p_idxs, q_idxs, axis=0)
        m = np.append (m, m+1)
    p_sorted = np.stack ([np.sort (prow) for prow in p_idxs], axis=0)
    idx_uniq = np.unique (p_sorted, return_index=True, axis=0)[1]
    p_idxs = p_idxs[idx_uniq]
    m = m[idx_uniq]
    return p_idxs, m

class FSUCCOperator (uccsd_sym0.FSUCCOperator):
    ''' Hide the spin degrees of freedom '''

    def __init__(self, norb, a_idxs, i_idxs):
        # Up to two equal indices in one generator are allowed
        # However, we still can't have any equal generators
        self.a_idxs = []
        self.i_idxs = []
        self.symtab = []
        for ix, (a, i) in enumerate (zip (a_idxs, i_idxs)):
            a = np.ascontiguousarray (a, dtype=np.uint8)
            i = np.ascontiguousarray (i, dtype=np.uint8)
            errstr = 'a,i={},{} invalid for number-sym op'.format (a,i)
            assert (len (a) == len (i)), errstr
            errstr = 'a,i={},{} degree of freedom undefined'.format (a,i)
            assert (not (np.all (a == i))), errstr
            if len (a) == 1: # Only case where I know the proper symmetry
                             # relation between amps to ensure S**2
                symrow = [len (self.a_idxs), len (self.i_idxs)+1]
                self.a_idxs.extend ([a, a+norb])
                self.i_idxs.extend ([i, i+norb])
                self.symtab.append (symrow)
            else:
                for ab, ma in zip (*spincases (a, norb)):
                    if np.amax (np.unique (ab, # nilpotent escape
                        return_counts=True)[1]) > 1: continue
                    for ij, mi in zip (*spincases (i, norb)):
                        if mi != ma: continue # sz-break escape
                        if np.amax (np.unique (ij, # nilpotent escape
                            return_counts=True)[1]) > 1: continue
                        self.symtab.append ([len (self.symtab)])
                        self.a_idxs.append (ab)
                        self.i_idxs.append (ij)
        self.norb = 2*norb
        self.ngen = len (self.a_idxs)
        assert (len (self.i_idxs) == self.ngen)
        self.ngen_uniq = len (self.symtab)
        self.uniq_gen_idx = np.array ([x[0] for x in self.symtab])
        self.amps = np.zeros (self.ngen)
        self.assert_sanity ()

    def assert_sanity (self):
        norb = self.norb // 2
        uccsd_sym0.FSUCCOperator.assert_sanity (self)
        for a, i in zip (self.a_idxs, self.i_idxs):
            errstr = 'a,i={},{} breaks sz symmetry'.format (a, i)
            assert (np.sum (a//norb) == np.sum (i//norb)), errstr

    def get_uniq_amps (self):
        return self.amps[self.uniq_gen_idx]

    def set_uniq_amps_(self, x):
        for symrow, xi in zip (self.symtab, x):
            self.amps[symrow] = xi
        return self

    def gen_deriv1 (self, psi, transpose=False):
        ''' Implement the product rule for constrained derivatives '''
        for symrow in self.symtab:
            dupsi = np.zeros_like (psi)
            for igend in symrow: dupsi += self.get_deriv1 (psi, igend,
                transpose=transpose)
            yield dupsi

    def print_tab (self):
        norb = self.norb // 2
        for ix in range (self.ngen_uniq): self.print_uniq (ix)

    def print_uniq (self, ix):
        symrow = self.symtab[ix]
        print ("Unique amplitude {}".format (ix))
        for gen in symrow:
            ab = self.a_idxs[gen]
            ij = self.i_idxs[gen]
            ptstr = "   {:12.5e} (".format (self.amps[gen])
            for i in ij: ptstr += "{}{}".format (i%norb,('a','b')[i//norb])
            ptstr += '->'
            for a in ab: ptstr += "{}{}".format (a%norb,('a','b')[a//norb])
            ptstr += ')'
            print (ptstr)

def get_uccs_op (norb, t1=None):
    t1_idx = np.tril_indices (norb, k=-1)
    a, i = list (t1_idx[0]), list (t1_idx[1])
    uop = FSUCCOperator (norb, a, i)
    if t1 is not None:
        uop.set_uniq_amps_(t1[t1_idx])
    return uop

def get_uccsd_op (norb, t1=None, t2=None):
    t1_idx = np.tril_indices (norb, k=-1)
    ab_idxs, ij_idxs = list (t1_idx[0]), list (t1_idx[1])
    pq = [(p, q) for p, q in zip (*np.tril_indices (norb))]
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
    x0 = uop.get_uniq_amps ()
    if t1 is not None: x0[:len (t1_idx[0])] = t1[t1_idx]
    if t2 is not None: raise NotImplementedError ("t2 initialization")
    uop.set_uniq_amps_(x0)
    return uop

def contract_s2 (psi, norb):
    assert (psi.size == 2**(2*norb))
    s2psi = np.zeros_like (psi)
    psi_ptr = psi.ctypes.data_as (ctypes.c_void_p)
    s2psi_ptr = s2psi.ctypes.data_as (ctypes.c_void_p)
    libfsucc.FSUCCcontractS2 (psi_ptr, s2psi_ptr,
        ctypes.c_uint (norb))
    return s2psi

def spin_square (psi, norb):
    ss = psi.dot (contract_s2 (psi, norb))
    s = np.sqrt (ss+0.25) - 0.5
    multip = s*2 + 1
    return ss, multip

class UCCS (uccsd_sym0.UCCS):
    def get_uop (self):
        return get_uccs_op (self.norb)

    def rotate_mo (self, mo_coeff=None, x=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if x is None: x=self.x
        norb = self.norb
        t1 = np.zeros ((norb, norb), dtype=x.dtype)
        t1[np.tril_indices (norb, k=-1)] = x[:]
        t1 -= t1.T
        umat = linalg.expm (t1)
        return mo_coeff @ umat


if __name__ == '__main__':
    norb = 4
    nelec = 4
    def pbin (n, k=norb):
        s = bin (n)[2:]
        m = (2*k) - len (s)
        if m: s = ''.join (['0',]*m) + s
        return s
    psi = np.zeros (2**(2*norb))
    psi[51] = 1.0

    from pyscf.fci import cistring, spin_op
    from mrh.exploratory.citools import fockspace

    t1_rand = np.random.rand (norb,norb)
    t2_rand = np.random.rand (norb,norb,norb,norb)
    uop_s = get_uccs_op (norb, t1=t1_rand)
    upsi = uop_s (psi)
    upsi_h = fockspace.fock2hilbert (upsi, norb, nelec)
    uTupsi = uop_s (upsi, transpose=True)
    for ix in range (2**(2*norb)):
        if np.any (np.abs ([psi[ix], upsi[ix], uTupsi[ix]]) > 1e-8):
            print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))
    print ("<psi|S**2|psi> =",spin_square (psi, norb)[0],
           "<psi|U'S**2U|psi> =",spin_square (upsi, norb)[0],spin_op.spin_square (upsi_h, norb, nelec)[0])

    uop_sd = get_uccsd_op (norb)
    x_rand = (1 - 2*np.random.rand (uop_sd.ngen_uniq)) * math.pi/4
    uop_sd.set_uniq_amps_(x_rand)
    upsi = uop_sd (psi)
    upsi_h = fockspace.fock2hilbert (upsi, norb, nelec)
    uTupsi = uop_sd (upsi, transpose=True)
    for ix in range (2**(2*norb)):
        if np.any (np.abs ([psi[ix], upsi[ix], uTupsi[ix]]) > 1e-8):
            print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))
    print ("<psi|S**2|psi> =",spin_square (psi, norb)[0],
           "<psi|U'S**2U|psi> =",spin_square (upsi, norb)[0], spin_op.spin_square (upsi_h, norb, nelec)[0])

    ndet = cistring.num_strings (norb, nelec//2)
    np.random.seed (0)
    tpsi = 1-(2*np.random.rand (ndet))
    tpsi = np.multiply.outer (tpsi, tpsi).ravel ()
    tpsi /= linalg.norm (tpsi)
    tpsi = fockspace.hilbert2fock (tpsi, norb, (nelec//2, nelec//2)).ravel ()
    from scipy import optimize
    def obj_test (uop_test):
        def obj_fun (x):
            uop_test.set_uniq_amps_(x)
            upsi = uop_test (psi)
            ut = upsi.dot (tpsi)
            err = upsi.dot (upsi) - (ut**2)
            jac = np.zeros_like (x)
            for ix, dupsi in enumerate (uop_test.gen_deriv1 (psi)):
                jac[ix] += 2*dupsi.dot (upsi - ut*tpsi) 
            print (err, linalg.norm (jac))
            return err, jac

        res = optimize.minimize (obj_fun, uop_test.get_uniq_amps (), method='BFGS', jac=True)

        print (res.success)
        uop_test.set_uniq_amps_(res.x)
        upsi = uop_test (psi)
        uTupsi = uop_test (upsi, transpose=True)
        for ix in range (2**(2*norb)):
            if np.any (np.abs ([psi[ix], upsi[ix], tpsi[ix]]) > 1e-8):
                print (pbin (ix), psi[ix], upsi[ix], tpsi[ix])
        print ("<psi|psi> =",psi.dot (psi), "<tpsi|psi> =",tpsi.dot (psi),"<tpsi|U|psi> =",tpsi.dot (upsi))
        print ("<psi|S**2|psi> =",spin_square (psi, norb)[0],
               "<psi|U'S**2U|psi> =",spin_square (upsi, norb)[0])

    uop_s.set_uniq_amps_(np.zeros (uop_s.ngen_uniq))
    print ("Testing singles...")
    obj_test (uop_s)
    print ('Testing singles and doubles...')
    x = np.zeros (uop_sd.ngen_uniq)
    #x[:uop_s.ngen_uniq] = uop_s.get_uniq_amps ()
    uop_sd.set_uniq_amps_(x)
    obj_test (uop_sd)

    from pyscf import gto, scf, lib
    mol = gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis='6-31g', verbose=lib.logger.DEBUG, output='uccsd_sym0.log')
    rhf = scf.RHF (mol).run ()
    uccs = UCCS (mol).run ()
    print ("The actual test result is:", uccs.e_tot-rhf.e_tot, linalg.norm (uccs.x))
    nmo = mol.nao_nr ()
    hf_mo = rhf.mo_coeff
    uccs_mo0 = uccs.mo_coeff
    uccs_mo1 = uccs.rotate_mo ()
    s0 = mol.intor_symmetric ('int1e_ovlp')
    print ("hf MOs vs UCCS frame:\n", np.diag (hf_mo.T @ s0 @ uccs_mo0))
    print ("hf MOs vs UCCS opt:\n", np.diag (hf_mo.T @ s0 @ uccs_mo1))
    rhf.mo_coeff[:,:] = uccs_mo1[:,:]
    print (rhf.energy_tot (), uccs.e_tot)


