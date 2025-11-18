import copy
import unittest
import numpy as np
from scipy import linalg
from mrh.my_pyscf.lassi import basis
from pyscf import lib
import itertools

def setUpModule():
    global rng
    rng = np.random.default_rng ()

def tearDownModule():
    global rng
    del rng

def iterate_spins (max_nfrag, max_smult_f, max_smult=None):
    for nfrag in range (1,max_nfrag+1):
        for smults in itertools.product (range (1,max_smult_f+1), repeat=nfrag):
            smult_f = np.asarray (smults)
            max_spin1 = (smult_f-1).sum ()
            if max_smult is None:
                max_spin = max_spin1
            else:
                max_spin = max_smult - 1
            if ((max_spin1-max_spin)%2) != 0:
                max_spin -= 1
            assert (max_spin >= 0)
            for spin in range (-max_spin, max_spin+1, 2):
                yield smult_f, spin  

class KnownValues(unittest.TestCase):

    def test_s2mat (self):
        for smult_f, spin in iterate_spins (4, 4):
            with self.subTest (smult_f=smult_f, spin=spin):
                s2mat = basis.make_s2mat (smult_f, spin)

if __name__ == "__main__":
    print("Full Tests for LASSI basis module spin functions")
    unittest.main()


