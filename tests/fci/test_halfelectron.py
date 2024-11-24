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
    if (not cre) and (not nelec[spin]): return
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

def case_contract_3he (self, norb, nelec, cre, spin, _incl_1he=True):
    if (not cre) and (not nelec[spin]): return
    ci = rng.random ((cistring.num_strings (norb,nelec[0]),
                         cistring.num_strings (norb,nelec[1])), dtype=float)
    ci /= linalg.norm (ci)
    ops = [[des_a, des_b], [cre_a, cre_b]]
    h1he = rng.random ((norb), dtype=float)
    h3he = rng.random ((norb,norb,norb), dtype=float)
    if cre:
        h3he = (h3he + h3he.transpose (0,2,1)) / 2
    else:
        h3he = (h3he + h3he.transpose (1,0,2)) / 2
    op = ops[int(cre)][spin]
    if not _incl_1he: h1he[:] = 0
    hci_ref = np.tensordot (h1he, np.stack ([op (ci, norb, nelec, i)
                                             for i in range (norb)],
                                            axis=0),
                            axes=1)
    nelecl = [nelec[0], nelec[1]]
    nelecl[spin] -= 1
    for i in range (norb):
        if cre:
            hci_ref += op (direct_spin1.contract_1e (h3he[i], ci, norb, nelec),
                           norb, nelec, i)
        else:
            hci_ref += direct_spin1.contract_1e (h3he[:,:,i], op (ci, norb, nelec, i),
                                                 norb, nelecl)
    h3heff = direct_halfelectron.absorb_h1he (h1he, h3he, cre, spin, norb, nelec, fac=.5)
    hci_test = direct_halfelectron.contract_3he (
        h3heff, cre, spin, ci, norb, nelec
    )
    self.assertAlmostEqual (lib.fp (hci_test), lib.fp (hci_ref), 8)
 
class KnownValues(unittest.TestCase):

    def test_contract_1he (self):
        for norb, nelec in cases:
            for cre in (True, False):
                for spin in range (2):
                    with self.subTest (cre=cre, spin=spin, norb=norb, nelec=nelec):
                        case_contract_1he (self, norb, nelec, cre, spin)

    def test_contract_3he (self):
        for norb, nelec in cases:
            for cre in (True, False):
                for spin in range (2):
                    with self.subTest (cre=cre, spin=spin, norb=norb, nelec=nelec):
                        case_contract_3he (self, norb, nelec, cre, spin, _incl_1he=False)

    def test_contract_13he (self):
        for norb, nelec in cases:
            for cre in (True, False):
                for spin in range (2):
                    with self.subTest (cre=cre, spin=spin, norb=norb, nelec=nelec):
                        case_contract_3he (self, norb, nelec, cre, spin)


if __name__ == "__main__":
    print("Full Tests for direct_halfelectron")
    unittest.main()

