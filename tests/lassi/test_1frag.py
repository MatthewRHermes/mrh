import copy
import unittest
import numpy as np
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.mcscf.addons import state_average_mix
from mrh.my_pyscf.fci import csf_solver
from mrh.tests.lasscf.me2n2_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def setUpModule():
    global mol, mf, mc_ss, mc_sa, mc_exc
    r_nn = 3.0
    mol = struct (3.0, '6-31g')
    mol.output = '/dev/null'
    mol.verbose = lib.logger.DEBUG
    mol.build ()
    mf = scf.RHF (mol).run ()
    mc_ss = mcscf.CASSCF (mf, 4, 4).run ()
    mc_sa = state_average_mix (mcscf.CASSCF (mf, 4, 4),
        [csf_solver (mol, smult=m2+1).set (spin=m2) for m2 in (0,2)],
        [0.5, 0.5]).run ()
    mc_exc = mcscf.CASCI (mf, 4, 4)
    mc_exc.fcisolver = csf_solver (mol, smult=1)
    mc_exc.fcisolver.nroots = 2
    mc_exc.mo_coeff = mc_ss.mo_coeff
    mc_exc.kernel ()

def tearDownModule():
    global mol, mf, mc_ss, mc_sa, mc_exc
    mol.stdout.close ()
    del mol, mf, mc_ss, mc_sa, mc_exc

class KnownValues(unittest.TestCase):

    def test_ss (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5).run ()
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        self.assertAlmostEqual (e_o0[0], mc_ss.e_tot, 7)
        self.assertAlmostEqual (e_o1[0], mc_ss.e_tot, 7)

    def test_sa (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5)
        las.state_average_(weights=[0.5,0.5], spins=[0,2]).run ()
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        for e_si0, e_si1, e_mc in zip (e_o0, e_o1, mc_sa.e_states):
            self.assertAlmostEqual (e_si0, e_mc, 7)
            self.assertAlmostEqual (e_si1, e_mc, 7)

    def test_exc (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5)
        las.mo_coeff = mc_exc.mo_coeff
        lroots = np.array ([[2]])
        las.lasci (lroots=lroots)
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        for e_si0, e_si1, e_mc in zip (e_o0, e_o1, mc_exc.e_tot):
            self.assertAlmostEqual (e_si0, e_mc, 7)
            self.assertAlmostEqual (e_si1, e_mc, 7)

if __name__ == "__main__":
    print("Full Tests for LASSI single-fragment edge case")
    unittest.main()
