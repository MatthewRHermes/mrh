import copy
import unittest
import numpy as np
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.mcscf.addons import state_average_mix
from mrh.my_pyscf.fci import csf_solver
from mrh.tests.lasscf.me2n2_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.lassi.sitools import make_sdm1
from mrh.my_pyscf.lassi.lassi import roots_trans_rdm12s
from mrh.my_pyscf.lassi.op_o1 import get_fdm1_maker
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_contract_op_si

op = (op_o0, op_o1)

def setUpModule():
    global mol, mf, mc_ss, mc_sa, mc_exc, las_exc, lsi
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
    las_exc = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5)
    las_exc.mo_coeff = mc_exc.mo_coeff
    lroots = np.array ([[2]])
    las_exc.lasci (lroots=lroots)
    lsi = lassi.LASSI (las_exc).run ()


def tearDownModule():
    global mol, mf, mc_ss, mc_sa, mc_exc, las_exc, lsi
    mol.stdout.close ()
    del mol, mf, mc_ss, mc_sa, mc_exc, las_exc, lsi

class KnownValues(unittest.TestCase):

    def test_ss (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5).run ()
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        lsi = lassi.LASSI (las).run (davidson_only=True)
        self.assertAlmostEqual (e_o0[0], mc_ss.e_tot, 7)
        self.assertAlmostEqual (e_o1[0], mc_ss.e_tot, 7)
        self.assertAlmostEqual (lsi.e_roots[0], mc_ss.e_tot, 7)

    def test_sa (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5)
        las.state_average_(weights=[0.5,0.5], spins=[0,2]).run ()
        nroots = len (mc_sa.e_states)
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        lsi = lassi.LASSI (las).run (davidson_only=True, nroots_si=nroots)
        for e_si0, e_si1, ed, e_mc in zip (e_o0, e_o1, lsi.e_roots, mc_sa.e_states):
            self.assertAlmostEqual (e_si0, e_mc, 7)
            self.assertAlmostEqual (e_si1, e_mc, 7)
            self.assertAlmostEqual (ed, e_mc, 7)

    def test_exc (self):
        las = las_exc
        nroots = len (mc_exc.e_tot)
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        lsi = lassi.LASSI (las).run (davidson_only=True, nroots_si=nroots)
        for e_si0, e_si1, ed, e_mc in zip (e_o0, e_o1, lsi.e_roots, mc_exc.e_tot):
            self.assertAlmostEqual (e_si0, e_mc, 7)
            self.assertAlmostEqual (e_si1, e_mc, 7)
            self.assertAlmostEqual (ed, e_mc, 7)

    def test_fdm1 (self):
        fdm1 = get_fdm1_maker (lsi, lsi.ci, lsi.get_nelec_frs (), lsi.si) (0,0)
        sdm1 = make_sdm1 (lsi, 0, 0)
        self.assertAlmostEqual (lib.fp (fdm1), lib.fp (sdm1), 7)
        fdm1[0,0,0] -= 1
        fdm1[1,1,1] -= 1
        self.assertLess (np.amax (np.abs (fdm1)), 1e-8)

    def test_rdms (self):
        si_bra = lsi.si
        si_ket = np.roll (lsi.si, 1, axis=1)
        d1_ref, d2_ref = roots_trans_rdm12s (lsi, lsi.ci, si_bra, si_ket, opt=0)
        for opt in range (1,2):
            d1_test, d2_test = roots_trans_rdm12s (lsi, lsi.ci, si_bra, si_ket, opt=opt)
            with self.subTest (opt=opt):
                self.assertAlmostEqual (lib.fp (d1_test), lib.fp (d1_ref), 7)
                self.assertAlmostEqual (lib.fp (d2_test), lib.fp (d2_ref), 7)

    def test_contract_hlas_ci (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        case_contract_hlas_ci (self, las, h0, h1, h2, las.ci, lsi.get_nelec_frs ())

    def test_contract_op_si (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        case_contract_op_si (self, las, h1, h2, las.ci, lsi.get_nelec_frs ())

if __name__ == "__main__":
    print("Full Tests for LASSI single-fragment edge case")
    unittest.main()
