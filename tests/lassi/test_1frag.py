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
from mrh.my_pyscf.lassi.lassi import roots_make_rdm12s
from mrh.my_pyscf.lassi.op_o1 import get_fdm1_maker
from mrh.my_pyscf.lassi import op_o0, op_o1

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
        las = las_exc
        e_o0,si_o0=las.lassi(opt=0)
        e_o1,si_o1=las.lassi(opt=1)
        for e_si0, e_si1, e_mc in zip (e_o0, e_o1, mc_exc.e_tot):
            self.assertAlmostEqual (e_si0, e_mc, 7)
            self.assertAlmostEqual (e_si1, e_mc, 7)

    def test_fdm1 (self):
        fdm1 = get_fdm1_maker (lsi, lsi.ci, lsi.get_nelec_frs (), lsi.si) (0,0)
        sdm1 = make_sdm1 (lsi, 0, 0)
        self.assertAlmostEqual (lib.fp (fdm1), lib.fp (sdm1), 7)
        fdm1[0,0,0] -= 1
        fdm1[1,1,1] -= 1
        self.assertLess (np.amax (np.abs (fdm1)), 1e-8)

    def test_rdms (self):
        d1_ref, d2_ref = roots_make_rdm12s (lsi, lsi.ci, lsi.si, opt=0)
        for opt in range (1,2):
            d1_test, d2_test = roots_make_rdm12s (lsi, lsi.ci, lsi.si, opt=opt)
            with self.subTest (opt=opt):
                self.assertAlmostEqual (lib.fp (d1_test), lib.fp (d1_ref), 7)
                self.assertAlmostEqual (lib.fp (d2_test), lib.fp (d2_ref), 7)

    def test_contract_hlas_ci (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        nelec = lsi.get_nelec_frs ()
        print ("huh?", nelec)
        ci_fr = las.ci
        ham = (si * (e_roots[None,:]-h0)) @ si.conj ().T
        ndim = len (e_roots)

        lroots = lsi.get_lroots ()
        lroots_prod = np.prod (lroots, axis=0)
        nj = np.cumsum (lroots_prod)
        ni = nj - lroots_prod
        for opt in range (2):
            hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, ci_fr, nelec)
            for f, (ci_r, hket_r_pabq) in enumerate (zip (ci_fr, hket_fr_pabq)):
                current_order = list (range (las.nfrags-1, -1, -1)) + [las.nfrags]
                current_order.insert (0, current_order.pop (f))
                for r, (ci, hket_pabq) in enumerate (zip (ci_r, hket_r_pabq)):
                    if ci.ndim < 3: ci = ci[None,:,:]
                    proper_shape = np.append (lroots[:,r], ndim)
                    current_shape = proper_shape[current_order]
                    to_proper_order = list (np.argsort (current_order))
                    hket_pq = lib.einsum ('rab,pabq->rpq', ci.conj (), hket_pabq)
                    hket_pq = hket_pq.reshape (current_shape)
                    hket_pq = hket_pq.transpose (*to_proper_order)
                    hket_pq = hket_pq.reshape ((lroots_prod[r], ndim))
                    hket_ref = ham[ni[r]:nj[r]]
                    for s, (k, l) in enumerate (zip (ni, nj)):
                        hket_pq_s = hket_pq[:,k:l]
                        hket_ref_s = hket_ref[:,k:l]
                        with self.subTest (opt=opt, frag=f, bra_space=r, ket_space=s):
                            print (opt, f, r, s)
                            print (hket_pq_s)
                            print (hket_ref_s)
                            self.assertAlmostEqual (lib.fp (hket_pq_s), lib.fp (hket_ref_s), 8)


if __name__ == "__main__":
    print("Full Tests for LASSI single-fragment edge case")
    unittest.main()
