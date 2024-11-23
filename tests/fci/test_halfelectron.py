import unittest
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from mrh.my_pyscf.fci import direct_halfelectron

def setUpModule ():
    global rng, cases
    cases = [(2, (1,1)),
             (2, (1,0)),
             (2, (0,1)),
             (3, (1,0)),
             (3, (0,1)),
             (3, (1,1)),
             (3, (2,0)),
             (3, (0,2)),
             (3, (2,1)),
             (3, (1,2)),
             (3, (2,2)),
             (5, (3,2))]
    rng = np.random.default_rng (0)

def tearDownModule ():
    global rng, cases
    del rng, cases

def case_contract_1he (self, norb, nelec, cre, spin):
    ci = rng.random ((cistring.num_strings (norb,nelec[0]),
                         cistring.num_strings (norb,nelec[1])), dtype=float)
    ci /= linalg.norm (ci)
    ops = [[des_a, des_b], [cre_a, cre_b]]
    h1he = rng.random ((norb), dtype=float)
    op = ops[int(cre)][spin]
    hci_ref = np.tensordot (h1he, np.stack ([op (ci, norb, nelec, i)
                                             for i in range (norb)],
                                            axis=0),
                            axes=1)
    hci_test = direct_halfelectron.contract_1he (
        h1he, cre, spin, ci, norb, nelec
    )
    self.assertAlmostEqual (lib.fp (hci_test), lib.fp (hci_ref), 8)
 
class KnownValues(unittest.TestCase):

    def test_contract_1he (self):
        for norb, nelec in cases:
            for cre in (True, False):
                for spin in range (2):
                    with self.subTest (cre=cre, spin=spin, norb=norb, nelec=nelec):
                        case_contract_1he (self, norb, nelec, cre, spin)
            break

if __name__ == "__main__":
    print("Full Tests for direct_halfelectron")
    unittest.main()

