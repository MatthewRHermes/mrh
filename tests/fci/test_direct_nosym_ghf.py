import unittest
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from mrh.my_pyscf.fci import direct_nosym_ghf
from itertools import product

def contract_1e_ref (h1e, ci, norb, nelec):
    hci = [0,0,0]
    for sc,sd in product (range (2), repeat=2):
        a = 1 + (sc-sd)
        if nelec[sd]==0: continue
        if (sc!=sd) and ((norb-nelec[sc])==0): continue
        cre_op = (cre_a,cre_b)[sc]
        des_op = (des_a,des_b)[sd]
        nelecq = list (nelec)
        nelecq[sd] = nelecq[sd]-1
        for q in range (norb):
            qci = des_op (ci, norb, nelec, q)
            j = (sd*norb) + q
            for p in range (norb):
                i = (sc*norb) + p
                hci[a] += h1e[i,j] * cre_op (qci, norb, nelecq, p)
    return hci

class KnownValues(unittest.TestCase):

    def test_contract_1e(self):
        for norb in range (3,6):
            h1e = np.random.rand (2*norb,2*norb)
            h1e[:] = 0
            h1e[0,0] = 1
            na, nb = norb - norb//2, norb//2
            for nelec in ((na,nb),(na+1,nb),(na,nb+1)):
                ndeta = cistring.num_strings (norb, nelec[0])
                ndetb = cistring.num_strings (norb, nelec[1])
                ci = np.random.rand (ndeta,ndetb)
                ci /= linalg.norm (ci)
                hci_test = direct_nosym_ghf.contract_1e (h1e, ci, norb, nelec)
                hci_ref = contract_1e_ref (h1e, ci, norb, nelec)
                for ht, hr in zip (hci_test, hci_ref):
                    self.assertAlmostEqual (lib.fp (ht), lib.fp (hr), 8)


if __name__ == "__main__":
    print("Full Tests for direct_nosym_ghf")
    unittest.main()

