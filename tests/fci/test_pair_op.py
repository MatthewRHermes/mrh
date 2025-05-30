import unittest
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from mrh.my_pyscf.fci import pair_op

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

def case_pair_op (self, norb, nelec, cre, spin):
    s1 = int (spin>1)
    s2 = int (spin>0)
    dneleca = (-1,1)[int(cre)] * (int (s1==0) + int (s2==0))
    dnelecb = (-1,1)[int(cre)] * (int (s1==1) + int (s2==1))
    dnelecq = (-1,1)[int(cre)]
    nelec1 = (nelec[0] + dneleca, nelec[1] + dnelecb)
    if nelec1[0] > norb or nelec1[1] > norb or nelec1[0] < 0 or nelec1[1] < 0: return
    ci = rng.random ((cistring.num_strings (norb,nelec[0]),
                      cistring.num_strings (norb,nelec[1])), dtype=float)
    ci /= linalg.norm (ci)
    ops = [[des_a, des_b], [cre_a, cre_b]]
    hpair = rng.random ((norb,norb), dtype=float)
    hci_ref = 0
    op1 = ops[cre][s1]
    op2 = ops[cre][s2]
    sq = (s1,s2)[int (cre)]
    nelecq = [nelec[0], nelec[1]]
    nelecq[sq] += (-1,1)[int(cre)]
    opq = (op1,op2)[int (cre)]
    opp = (op2,op1)[int (cre)]
    for q in range (norb):
        qci = opq (ci, norb, nelec, q)
        for p in range (norb):
            hci_ref += hpair[p,q] * opp (qci, norb, nelecq, p)
    hci_test = pair_op.contract_pair_op (
        hpair, cre, spin, ci, norb, nelec
    )
    self.assertAlmostEqual (lib.fp (hci_test), lib.fp (hci_ref), 8)

class KnownValues(unittest.TestCase):

    def test_pair_op (self):
        for norb, nelec in cases:
            for cre in (True, False):
                for spin in range (3):
                    with self.subTest (cre=cre, spin=spin, norb=norb, nelec=nelec):
                        case_pair_op (self, norb, nelec, cre, spin)

if __name__ == "__main__":
    print("Full Tests for pair_op")
    unittest.main()

