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


def possible_smults (smult_f, spin):
    s2_f = smult_f-1
    s2_f = s2_f[s2_f!=0]
    if len (s2_f) == 0: return np.arange (1,2,dtype=int)
    s2_f = np.sort (s2_f)[::-1]
    smult_min = max (2*s2_f[0] - s2_f.sum (), abs (spin)) + 1
    smult_max = (smult_f-1).sum () + 1
    return np.arange (smult_min, smult_max+1, 2, dtype=int)

class KnownValues(unittest.TestCase):
    def iterate_spins (self, max_nfrag, max_smult_f, max_smult=None):
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
                    with self.subTest (smult_f=smult_f, spin=spin):
                        yield smult_f, spin

    def check_s2mat (self, s2mat, smult_f, spin):
        with self.subTest ('check_s2mat'):
            evals = linalg.eigh (s2mat)[0]
            idx = np.zeros (evals.shape[0], dtype=bool)
            for smult in possible_smults (smult_f, spin):
                s = (smult-1) * .5
                s2eval = s * (s + 1)
                idx[np.isclose (evals, s2eval)] = True
            self.assertTrue (np.all (idx))

    def check_umat (self, umat):
        with self.subTest ('check_umat'):
            ovlp = umat.conj ().T @ umat
            ref = np.eye (umat.shape[1])
            self.assertAlmostEqual (lib.fp (ovlp), lib.fp (ref), 7)

    def check_umat_s2mat (self, umat, s2mat, smult):
        with self.subTest ('check_umat_s2mat'):
            s = (smult-1) / 2
            s2eval = s * (s + 1)
            ref = np.eye (umat.shape[1]) * s2eval
            test = umat.conj ().T @ s2mat @ umat
            self.assertAlmostEqual (lib.fp (test), lib.fp (ref), 7)

    def test_spincoup (self):
        for smult_f, spin in self.iterate_spins (4, 4):
            s2mat = basis.make_s2mat (smult_f, spin)
            self.check_s2mat (s2mat, smult_f, spin)
            for smult in possible_smults (smult_f, spin):
                with self.subTest ('umat', smult_lsf=smult):
                    umat = basis.get_spincoup_umat (smult_f, spin, smult)
                    self.check_umat (umat)
                    self.check_umat_s2mat (umat, s2mat, smult)

if __name__ == "__main__":
    print("Full Tests for LASSI basis module spin functions")
    unittest.main()


