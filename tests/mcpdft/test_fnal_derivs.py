import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.mcpdft.pdft_feff import EotOrbitalHessianOperator
from mrh.my_pyscf.mcpdft.pdft_feff import vector_error
import unittest

h2 = scf.RHF (gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = '6-31g', 
    output='/dev/null')).run ()
lih = scf.RHF (gto.M (atom = 'Li 0 0 0; H 1.2 0 0', basis = 'sto-3g',
    output='/dev/null')).run ()

def case (kv, mc, mol, state, fnal):
    hop = EotOrbitalHessianOperator (mc, incl_d2rho=True)
    x0 = -hop.g_orb / hop.h_diag
    x0[hop.g_orb==0] = 0
    err_tab = np.zeros ((2,4))
    for ix, p in enumerate (range (17,19)):
        # For numerically unstable (i.e., translated) fnals,
        # it is somewhat difficult to find the convergence plateau
        # However, repeated calculations should show that
        # failure is rare and due only to numerical instability
        # and chance.
        x1 = x0 / (2**p)
        x1_norm = linalg.norm (x1)
        dg_test, de_test = hop (x1)
        dg_ref, de_ref = hop.seminum_orb (x1)
        de_err = abs ((de_test-de_ref)/de_ref)
        dg_err, dg_theta = vector_error (dg_test, dg_ref)
        err_tab[ix,:] = [x1_norm, de_err, dg_err, dg_theta]
    conv_tab = err_tab[1:,:] / err_tab[:-1,:]
    with kv.subTest (q='x'):
        kv.assertAlmostEqual (conv_tab[-1,0], 0.5, 9)
    with kv.subTest (q='de'):
        kv.assertLessEqual (err_tab[-1,1], 1e-4)
        kv.assertAlmostEqual (conv_tab[-1,1], 0.5, 2)
    with kv.subTest (q='d2e'):
        kv.assertLessEqual (err_tab[-1,2], 1e-4)
        kv.assertAlmostEqual (conv_tab[-1,2], 0.5, 2)
    

def tearDownModule():
    global h2, lih
    h2.mol.stdout.close ()
    lih.mol.stdout.close ()
    del h2, lih

class KnownValues(unittest.TestCase):

    def test_de_d2e (self):
        for mol, mf in zip (('H2', 'LiH'), (h2, lih)):
            for state, nel in zip (('Singlet', 'Triplet'), (2, (2,0))):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE'):
                    mc = mcpdft.CASSCF (mf, fnal, 2, nel).run ()
                    with self.subTest (mol=mol, state=state, fnal=fnal):
                        case (self, mc, mol, state, fnal)

if __name__ == "__main__":
    print("Full Tests for MC-PDFT fnal derivatives")
    unittest.main()






