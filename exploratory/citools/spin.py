import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.spin_op import spin_square0
from pyscf.fci.addons import _unpack_nelec
import itertools

def spin_square_diag (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    occslsta = occslstb = cistring._gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslstb = cistring._gen_occslst(range(norb), nelecb)
    xorlist = np.array ([[len (np.setxor1d (a, b)) for b in occslstb] for a in occslsta])
    sz = (neleca-nelecb)*.5
    sdiag = sz*sz + xorlist*.5
    return sdiag

def spin_square_diag_check (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    na = cistring.num_strings (norb, neleca)
    nb = cistring.num_strings (norb, nelecb)
    sdiag = np.zeros ((na,nb))
    for ia in range (na):
        for ib in range (nb):
            ci0 = np.zeros_like (sdiag)
            ci0[ia,ib] = 1.0
            sdiag[ia,ib] = spin_square0 (ci0, norb, nelec) [0]
    return sdiag

if __name__=='__main__':
    for norb in range (6):
        for neleca in range (norb):
            for nelecb in range (norb):
                sdiag_test = spin_square_diag (norb, (neleca,nelecb))
                sdiag_ref = spin_square_diag_check (norb, (neleca,nelecb))
                print (norb, (neleca, nelecb), np.amax (np.abs (sdiag_test-sdiag_ref)), linalg.norm (sdiag_test))

