import numpy as np
from scipy import linalg
from mrh.exploratory.unitary_cc import uccsd_sym0
from itertools import combinations, permutations, product, combinations_with_replacement

# Enforce n, sz, and s**2 symmetry in unitary generators

class FSUCCOperator (uccsd_sym0.FSUCCOperator):
    ''' Hide the spin degrees of freedom '''

    def __init__(self, norb, a_idxs, i_idxs):
        # Up to two equal indices in one generator are allowed
        self.a_idxs = []
        self.i_idxs = []
        self.symtab = []
        for ix, (a, i) in enumerate (zip (a_idxs, i_idxs)):
            a = np.ascontiguousarray (a, dtype=np.uint8)
            i = np.ascontiguousarray (i, dtype=np.uint8)
            errstr = 'a,i={},{} invalid for number-sym op'.format (a,i)
            assert (len (a) == len (i)), errstr
            ai = np.stack ([a, i], axis=0)[None,:,:]
            symrow = []
            for ielec in range (ai.shape[2]):
                bj = ai.copy ()
                bj[:,:,ielec] += norb
                ai = np.append (ai, bj, axis=0)
            for bj in ai:
                b, j = bj[0,:], bj[1,:]
                if np.amax (np.unique (b, # nilpotent escape
                    return_counts=True)[1]) > 1: continue
                if np.amax (np.unique (j, # nilpotent escape
                    return_counts=True)[1]) > 1: continue
                # undefined escape
                if np.all (np.sort (b) == np.sort (j)): continue
                symrow.append (len (self.a_idxs))
                self.a_idxs.append (bj[0,:])
                self.i_idxs.append (bj[1,:])
            if len (symrow): self.symtab.append (symrow)
        self.norb = 2*norb
        self.ngen = len (self.a_idxs)
        assert (len (self.i_idxs) == self.ngen)
        self.ngen_uniq = len (self.symtab)
        self.uniq_gen_idx = np.array ([x[0] for x in self.symtab])
        self.amps = np.zeros (self.ngen)
        self.assert_sanity (nodupes=False)

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
    for ai, bj in combinations_with_replacement (pq, 2):
        if ((ai[0] == ai[1]) and (bj[0] == bj[1])): continue # escape undefined
        ab_idxs.append ([ai[0], bj[0]])
        ij_idxs.append ([ai[1], bj[1]])
        a.append (ai[0])
        b.append (bj[0])
        i.append (ai[1])
        j.append (bj[1])
        if ((ai[0] != ai[1]) and (bj[0] != bj[1])): # Twist!
            ab_idxs.append ([ai[0], bj[1]])
            ij_idxs.append ([ai[1], bj[0]])
            a.append (ai[0])
            b.append (bj[1])
            i.append (ai[1])
            j.append (bj[0])
    uop = FSUCCOperator (norb, ab_idxs, ij_idxs)
    x0 = uop.get_uniq_amps ()
    if t1 is not None: x0[:len (t1_idx[0])] = t1[t1_idx]
    if t2 is not None: x0[len (t1_idx[0]):] = t2[(a,i,b,j)]
    uop.set_uniq_amps_(x0)
    return uop

class UCCS (uccsd_sym0.UCCS):
    def get_uop (self):
        return get_uccs_op (self.norb)

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

    t1_rand = np.random.rand (norb,norb)
    t2_rand = np.random.rand (norb,norb,norb,norb)
    uop_s = get_uccs_op (norb, t1=t1_rand)
    upsi = uop_s (psi)
    uTupsi = uop_s (upsi, transpose=True)
    for ix in range (2**(2*norb)):
        if np.any (np.abs ([psi[ix], upsi[ix], uTupsi[ix]]) > 1e-8):
            print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))

    uop_sd = get_uccsd_op (norb, t1=t1_rand, t2=t2_rand)
    upsi = uop_sd (psi)
    uTupsi = uop_sd (upsi, transpose=True)
    for ix in range (2**(2*norb)):
        if np.any (np.abs ([psi[ix], upsi[ix], uTupsi[ix]]) > 1e-8):
            print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))

    from pyscf.fci import cistring
    from mrh.exploratory.citools import fockspace
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
        print (uop_test.amps)

    uop_s.set_uniq_amps_(np.zeros (uop_s.ngen_uniq))
    print ("Testing singles...")
    obj_test (uop_s)
    print ('Testing singles and doubles...')
    x = np.zeros (uop_sd.ngen_uniq)
    x[:uop_s.ngen_uniq] = uop_s.get_uniq_amps ()
    #uop_sd.set_uniq_amps_(x)
    obj_test (uop_sd)


