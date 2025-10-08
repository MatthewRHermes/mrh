import unittest
import numpy as np
import itertools
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from mrh.my_pyscf.fci import rdm

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

def case_trans_rdm1hs (self, norb, nelec):
    ciket = 1 - 2*(rng.random ((cistring.num_strings (norb,nelec[0]),
                                cistring.num_strings (norb,nelec[1])), dtype=float))
    ciket /= linalg.norm (ciket)
    cibra_alpha = 1 - 2*(rng.random ((cistring.num_strings (norb,nelec[0]+1),
                                      cistring.num_strings (norb,nelec[1])), dtype=float))
    cibra_alpha /= linalg.norm (cibra_alpha)
    cibra_beta = 1 - 2*(rng.random ((cistring.num_strings (norb,nelec[0]),
                                     cistring.num_strings (norb,nelec[1]+1)), dtype=float))
    cibra_beta /= linalg.norm (cibra_beta)
    tdm1ha_ref = [np.dot (cibra_alpha.conj ().flat, cre_a (ciket, norb, nelec, i).flat)
                  for i in range (norb)]
    tdm1hb_ref = [np.dot (cibra_beta.conj ().flat, cre_b (ciket, norb, nelec, i).flat)
                  for i in range (norb)]
    tdm1ha = rdm.trans_rdm1ha_cre (cibra_alpha, ciket, norb, nelec)
    tdm1hb = rdm.trans_rdm1hb_cre (cibra_beta, ciket, norb, nelec)
    self.assertAlmostEqual (lib.fp (tdm1ha), lib.fp (tdm1ha_ref), 8)
    self.assertAlmostEqual (lib.fp (tdm1hb), lib.fp (tdm1hb_ref), 8)

def case_trans_rdm13hs (self, norb, nelec):
    ciket = rng.random ((cistring.num_strings (norb,nelec[0]),
                         cistring.num_strings (norb,nelec[1])), dtype=float)
    ciket /= linalg.norm (ciket)
    cibra_alpha = rng.random ((cistring.num_strings (norb,nelec[0]+1),
                               cistring.num_strings (norb,nelec[1])), dtype=float)
    cibra_alpha /= linalg.norm (cibra_alpha)
    cibra_beta = rng.random ((cistring.num_strings (norb,nelec[0]),
                              cistring.num_strings (norb,nelec[1]+1)), dtype=float)
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
    tdm1ha, (tdm3haa, tdm3hab) = rdm.trans_rdm13ha_cre (cibra_alpha, ciket, norb, nelec)
    self.assertAlmostEqual (lib.fp (tdm1ha), lib.fp (tdm1ha_ref), 8)
    self.assertAlmostEqual (lib.fp (tdm3haa), lib.fp (tdm3haa_ref), 8)
    self.assertAlmostEqual (lib.fp (tdm3hab), lib.fp (tdm3hab_ref), 8)
    tdm1hb, (tdm3hba, tdm3hbb) = rdm.trans_rdm13hb_cre (cibra_beta, ciket, norb, nelec)
    self.assertAlmostEqual (lib.fp (tdm1hb), lib.fp (tdm1hb_ref), 8)
    self.assertAlmostEqual (lib.fp (tdm3hba), lib.fp (tdm3hba_ref), 8)
    self.assertAlmostEqual (lib.fp (tdm3hbb), lib.fp (tdm3hbb_ref), 8)

def case_trans_sfudm1 (self, norb, nelec):
    if not nelec[1]: return
    ciket = rng.random ((cistring.num_strings (norb,nelec[0]),
                         cistring.num_strings (norb,nelec[1])), dtype=float)
    ciket /= linalg.norm (ciket)
    cibra = rng.random ((cistring.num_strings (norb,nelec[0]+1),
                         cistring.num_strings (norb,nelec[1]-1)), dtype=float)
    cibra /= linalg.norm (cibra)
    nelec_ket = nelec
    nelec_bra = (nelec[0]+1, nelec[1]-1)
    pbra = np.stack ([des_a (cibra, norb, nelec_bra, i) for i in range (norb)], axis=0)
    qket = np.stack ([des_b (ciket, norb, nelec_ket, i) for i in range (norb)], axis=0)
    sfudm1_ref = np.dot (pbra.reshape (norb,-1).conj (), qket.reshape (norb,-1).T)
    sfudm1 = rdm.trans_sfudm1 (cibra, ciket, norb, nelec)
    self.assertAlmostEqual (lib.fp (sfudm1), lib.fp (sfudm1_ref), 8)

def case_trans_ppdm (self, norb, nelec, spin):
    if (norb-nelec[0]) < (2-spin): return
    if (norb-nelec[1]) < spin: return
    ciket = rng.random ((cistring.num_strings (norb,nelec[0]),
                         cistring.num_strings (norb,nelec[1])), dtype=float)
    ciket /= linalg.norm (ciket)
    cibra = rng.random ((cistring.num_strings (norb,nelec[0]+2-spin),
                         cistring.num_strings (norb,nelec[1]+spin)), dtype=float)
    cibra /= linalg.norm (cibra)
    nelec_ket = nelec
    nelec_bra = (nelec[0]+2-spin, nelec[1]+spin)
    op_ket = (cre_a, cre_b)[int (spin>0)]
    op_bra = (des_a, des_b)[int (spin>1)]
    pbra = np.stack ([op_bra (cibra, norb, nelec_bra, i) for i in range (norb)], axis=0)
    qket = np.stack ([op_ket (ciket, norb, nelec_ket, i) for i in range (norb)], axis=0)
    ppdm_ref = np.dot (pbra.reshape (norb,-1).conj (), qket.reshape (norb,-1).T)
    ppdm = rdm.trans_ppdm (cibra, ciket, norb, nelec, spin=spin)
    if abs (lib.fp (ppdm) - lib.fp (ppdm_ref)) > 1e-8:
        for i,j in itertools.product (range (norb), repeat=2):
            print (i,j,ppdm[i,j],ppdm_ref[i,j])
    self.assertAlmostEqual (lib.fp (ppdm), lib.fp (ppdm_ref), 8)

class KnownValues(unittest.TestCase):

    def test_trans_rdm13hs (self):
        for norb, nelec in cases:
            with self.subTest (norb=norb, nelec=nelec):
                case_trans_rdm13hs (self, norb, nelec)

    def test_trans_rdm1hs (self):
        for norb, nelec in cases:
            with self.subTest (norb=norb, nelec=nelec):
                case_trans_rdm1hs (self, norb, nelec)

    def test_trans_sfudm1 (self):
        for norb, nelec in cases:
            with self.subTest (norb=norb, nelec=nelec):
                case_trans_sfudm1 (self, norb, nelec)

    def test_trans_ppdm (self):
        for norb, nelec in cases:
            for spin in range (3):
                with self.subTest (norb=norb, nelec=nelec, spin=spin):
                    case_trans_ppdm (self, norb, nelec, spin)

if __name__ == "__main__":
    print("Full Tests for rdm")
    unittest.main()

