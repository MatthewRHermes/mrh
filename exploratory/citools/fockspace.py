import numpy as np
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from itertools import product

def hilbert2fock (ci, norb, nelec):
    nelec = _unpack_nelec (nelec)
    ndeta = cistring.num_strings (norb, nelec[0])
    ndetb = cistring.num_strings (norb, nelec[1])
    ci = np.asarray (ci).reshape (-1, ndeta, ndetb)
    nroots = ci.shape[0]
    ci1 = np.zeros ((nroots, 2**norb, 2**norb), dtype=ci.dtype)
    strsa = cistring.addrs2str (norb, nelec[0], list(range(ndeta)))
    strsb = cistring.addrs2str (norb, nelec[1], list(range(ndetb)))
    ci1[:,strsa[:,None],strsb[:]] = ci[:,:,:]
    return ci1

def fock2hilbert (ci, norb, nelec):
    nelec = _unpack_nelec (nelec)
    ci = np.asarray (ci).reshape (-1, 2**norb, 2**norb)
    nroots = ci.shape[0]
    ndeta = cistring.num_strings (norb, nelec[0])
    ndetb = cistring.num_strings (norb, nelec[1])
    ci1 = np.empty ((nroots, ndeta, ndetb), dtype=ci.dtype)
    strsa = cistring.addrs2str (norb, nelec[0], list(range(ndeta)))
    strsb = cistring.addrs2str (norb, nelec[1], list(range(ndetb)))
    ci1[:,:,:] = ci[:,strsa[:,None],strsb]
    return ci1

if __name__ == '__main__':
    norb = 8
    ci_f = np.random.rand (3, 256*256)
    s = np.diag (np.dot (ci_f.conj (), ci_f.T))
    ci_f /= np.sqrt (s)[:,None]
    ci_f = ci_f.reshape (3, 256, 256)
    ci_fhf = ci_f.copy ()
    norms = []
    neleca_avg = []
    nelecb_avg = []
    for (neleca, nelecb) in product (list (range (norb+1)), repeat=2):
        ci_h = fock2hilbert (ci_f, norb, (neleca, nelecb))
        n = (ci_h.conj () * ci_h).sum ((1,2))
        print ("na =",neleca,", nb=",nelecb,"subspace has shape =",ci_h.shape,"and weights =",n)
        norms.append (n)
        neleca_avg.append (n*neleca)
        nelecb_avg.append (n*nelecb)
        ci_fhf -= hilbert2fock (ci_h, norb, (neleca, nelecb))
    print ("This should be zero:",np.amax (np.abs (ci_fhf)))
    norms = np.stack (norms, axis=0).sum (0)
    print ("These should be ones:", norms)
    neleca_avg = np.stack (neleca_avg, axis=0).sum (0)
    nelecb_avg = np.stack (nelecb_avg, axis=0).sum (0)
    print ("<neleca>, <nelecb> =", neleca_avg, nelecb_avg)

