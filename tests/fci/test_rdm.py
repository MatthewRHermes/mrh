import unittest
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from mrh.my_pyscf.fci import rdm

class KnownValues(unittest.TestCase):

    def test_trans_rdm13hs (self):
        norb = 5
        nelec = (3,2)
        ciket = np.random.rand (cistring.num_strings (norb,nelec[0]),
                                cistring.num_strings (norb,nelec[1]))
        ciket /= linalg.norm (ciket)
        cibra_alpha = np.random.rand (cistring.num_strings (norb,nelec[0]+1),
                                      cistring.num_strings (norb,nelec[1]))
        cibra_alpha /= linalg.norm (cibra_alpha)
        cibra_beta = np.random.rand (cistring.num_strings (norb,nelec[0]),
                                     cistring.num_strings (norb,nelec[1]+1))
        cibra_beta /= linalg.norm (cibra_beta)
        tdm1ha_ref = np.zeros (norb, dtype=ciket.dtype)
        tdm1hb_ref = np.zeros (norb, dtype=ciket.dtype)
        tdm3haa_ref = np.zeros ((norb,norb,norb), dtype=ciket.dtype)
        tdm3hab_ref = np.zeros ((norb,norb,norb), dtype=ciket.dtype)
        tdm3hba_ref = np.zeros ((norb,norb,norb), dtype=ciket.dtype)
        tdm3hbb_ref = np.zeros ((norb,norb,norb), dtype=ciket.dtype)
        for i in range (norb):
            pcibra = des_a (cibra_alpha, norb, (nelec[0]+1,nelec[1]), i)
            tdm1ha_ref[i] = np.dot (pcibra.flat, ciket.flat)
            dm1a, dm1b = direct_spin1.trans_rdm1s (pcibra, ciket, norb, nelec)
            tdm3haa_ref[i,:,:] = dm1a.T
            tdm3hab_ref[i,:,:] = dm1b.T
            pcibra = des_b (cibra_beta, norb, (nelec[0],nelec[1]+1), i)
            tdm1hb_ref[i] = np.dot (pcibra.flat, ciket.flat)
            dm1a, dm1b = direct_spin1.trans_rdm1s (pcibra, ciket, norb, nelec)
            tdm3hba_ref[i,:,:] = dm1a.T
            tdm3hbb_ref[i,:,:] = dm1b.T
        tdm1hs, tdm3hs = rdm.trans_rdm13hs (cibra_alpha, cibra_beta, ciket, norb, nelec)
        tdm1ha, tdm1hb = tdm1hs
        tdm3haa, tdm3hab, tdm3hba, tdm3hbb = tdm3hs
        self.assertAlmostEqual (lib.fp (tdm1ha), lib.fp (tdm1ha_ref), 8)
        self.assertAlmostEqual (lib.fp (tdm1hb), lib.fp (tdm1hb_ref), 8)
        self.assertAlmostEqual (lib.fp (tdm3haa), lib.fp (tdm3haa_ref), 8)
        self.assertAlmostEqual (lib.fp (tdm3hab), lib.fp (tdm3hab_ref), 8)
        self.assertAlmostEqual (lib.fp (tdm3hba), lib.fp (tdm3hba_ref), 8)
        self.assertAlmostEqual (lib.fp (tdm3hbb), lib.fp (tdm3hbb_ref), 8)

    def test_trans_rdm1hs (self):
        norb = 5
        nelec = (3,2)
        ciket = np.random.rand (cistring.num_strings (norb,nelec[0]),
                                cistring.num_strings (norb,nelec[1]))
        ciket /= linalg.norm (ciket)
        cibra_alpha = np.random.rand (cistring.num_strings (norb,nelec[0]+1),
                                      cistring.num_strings (norb,nelec[1]))
        cibra_alpha /= linalg.norm (cibra_alpha)
        cibra_beta = np.random.rand (cistring.num_strings (norb,nelec[0]),
                                     cistring.num_strings (norb,nelec[1]+1))
        cibra_beta /= linalg.norm (cibra_beta)
        tdm1ha_ref = [np.dot (cibra_alpha.conj ().flat, cre_a (ciket, norb, nelec, i).flat)
                      for i in range (norb)]
        tdm1hb_ref = [np.dot (cibra_beta.conj ().flat, cre_b (ciket, norb, nelec, i).flat)
                      for i in range (norb)]
        tdm1ha, tdm1b = rdm.trans_rdm1hs (cibra_alpha, cibra_beta, ciket, norb, nelec)
        self.assertAlmostEqual (lib.fp (tdm1ha), lib.fp (tdm1ha_ref), 8)
        self.assertAlmostEqual (lib.fp (tdm1hb), lib.fp (tdm1hb_ref), 8)
        # None behavior
        tdm1ha, _ = rdm.trans_rdm1hs (cibra_alpha, None, ciket, norb, nelec)
        self.assertAlmostEqual (lib.fp (tdm1ha), lib.fp (tdm1ha_ref), 8)
        _, tdm1hb = rdm.trans_rdm1hs (None, cibra_beta, ciket, norb, nelec)
        self.assertAlmostEqual (lib.fp (tdm1hb), lib.fp (tdm1hb_ref), 8)


if __name__ == "__main__":
    print("Full Tests for direct_nosym_uhf")
    unittest.main()

