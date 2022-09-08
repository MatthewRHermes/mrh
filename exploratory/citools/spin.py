import numpy as np
from pyscf import lib
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.spin_op import spin_square0
from pyscf.fci.addons import _unpack_nelec
import itertools

def spin_square_diag (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    strsa = strsb = cistring.make_strings (range(norb), neleca)
    na = nb = len (strsa)
    strsa_uint8 = strsb_uint8 = strsa.view ('uint8').reshape (na, -1).transpose (1,0)
    if neleca != nelecb:
        strsb = cistring.make_strings (range(norb), nelecb)
        nb = len (strsb)
        strsb_uint8 = strsb.view ('uint8').reshape (nb, -1).transpose (1,0)
    sdiag = np.zeros ((na, nb), dtype=np.uint8)
    for ix, (taba, tabb) in enumerate (zip (strsa_uint8, strsb_uint8)):
        if (ix*8) > norb: break
        tab = np.bitwise_xor.outer (taba, tabb)
        for j in range (8):
            tab, acc = np.divmod (tab, 2)
            sdiag += acc
    tab = taba = tabb = strsa = strsb = strsa_uint = strsb_uint = None
    sdiag = sdiag.astype ('float64')*.5
    sz = (neleca-nelecb)*.5
    sdiag += sz*sz
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
    for norb in range (65):
        for neleca in range (norb):
            for nelecb in range (norb):
                if norb > 6: continue 
                #if neleca != 8: continue
                #if nelecb != 8: continue
                sdiag_test = spin_square_diag (norb, (neleca,nelecb))
                sdiag_ref = spin_square_diag_check (norb, (neleca,nelecb))
                print (norb, (neleca, nelecb), np.amax (np.abs (sdiag_test-sdiag_ref)), linalg.norm (sdiag_test))

