import unittest
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from mrh.my_pyscf.fci import direct_nosym_uhf

def contract_1e_ref (h1e, ci, norb, nelec):
    hci = 0
    for spin in range (2):
        if nelec[spin]==0: continue
        cre_op = (cre_a,cre_b)[spin]
        des_op = (des_a,des_b)[spin]
        nelecq = list (nelec)
        nelecq[spin] = nelecq[spin]-1
        for q in range (norb):
            qci = des_op (ci, norb, nelec, q)
            for p in range (norb):
                hci += h1e[spin,p,q] * cre_op (qci, norb, nelecq, p)
    return hci

class KnownValues(unittest.TestCase):

    def test_contract_1e(self):
        for norb in range (3,6):
            h1e = np.random.rand (2,norb,norb)
            na, nb = norb - norb//2, norb//2
            for nelec in ((na,nb),(na+1,nb),(na,nb+1)):
                ndeta = cistring.num_strings (norb, nelec[0])
                ndetb = cistring.num_strings (norb, nelec[1])
                ci = np.random.rand (ndeta,ndetb)
                ci /= linalg.norm (ci)
                hci_test = direct_nosym_uhf.contract_1e (h1e, ci, norb, nelec)
                hci_ref = contract_1e_ref (h1e, ci, norb, nelec)
                self.assertAlmostEqual (lib.fp (hci_test), lib.fp (hci_ref), 8)


if __name__ == "__main__":
    print("Full Tests for direct_nosym_uhf")
    unittest.main()

