#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf, mcscf, ao2mo
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, LASSIrq, LASSIrqCT
from mrh.my_pyscf.lassi.lassis import coords, grad_orb_ci_si
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, roots_trans_rdm12s, make_stdm12s
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi import op_o0, op_o1, lassis
from mrh.my_pyscf.lassi.op_o1 import get_fdm1_maker
from mrh.my_pyscf.lassi.sitools import make_sdm1
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_lassis_fbf_2_model_state
from mrh.tests.lassi.addons import case_lassis_fbfdm, case_contract_op_si, debug_contract_op_si
from mrh.tests.lassi.addons import case_lassis_grads, case_lassis_hessian, case_lassis_ugg
from mrh.tests.lassi.addons import eri_sector_indexes

def setUpModule ():
    global mol, mf, lsi, las, mc, op, lsis
    xyz='''H 0 0 0
    H 1 0 0
    H 3 0 0
    H 4 0 0'''
    mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=5, output='debug_22.log')
    mf = scf.RHF (mol).run ()

    # Random Hamiltonian
    rng = np.random.default_rng (424)
    mf._eri = rng.random (mf._eri.shape)
    hcore = rng.random ((4,4))
    hcore = hcore + hcore.T
    mf.get_hcore = lambda *args: hcore

    # LASSCF with CASCI-limit model space
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    las.lasci ()
    las1 = las
    for i in range (2): las1 = all_single_excitations (las1)
    charges, spins, smults, wfnsyms = get_space_info (las1)
    lroots = 4 - smults
    idx = (charges!=0) & (lroots==3)
    lroots[idx] = 1
    las1.conv_tol_grad = las.conv_tol_self = 9e99
    las1.lasci (lroots=lroots.T)
    las1.dump_spaces ()
    lsi = LASSI (las1)
    lsi.kernel (opt=0)

    # CASCI limit
    mc = mcscf.CASCI (mf, 4, 4).run ()

    op = (op_o0, op_o1)

    # LASSIS
    lsis = lassis.LASSIS (las).run ()

def tearDownModule():
    global mol, mf, lsi, las, mc, op, lsis
    mol.stdout.close ()
    del mol, mf, lsi, las, mc, op, lsis

class KnownValues(unittest.TestCase):

    @unittest.skip("debugging")
    def test_op_o1 (self):
        e_roots, si = LASSI (lsi._las).kernel (opt=1)
        self.assertAlmostEqual (lib.fp (e_roots), lib.fp (lsi.e_roots), 8)
        ovlp = si.conj ().T @ lsi.si
        u, svals, vh = linalg.svd (ovlp)
        self.assertAlmostEqual (lib.fp (svals), lib.fp (np.ones (len (svals))), 8)

    @unittest.skip("debugging")
    def test_casci_limit (self):
        # CASCI limit
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)

        # LASSI in the CASCI limit
        las, e_roots, si = lsi._las, lsi.e_roots, lsi.si
        with self.subTest ("total energy"):
            self.assertAlmostEqual (e_roots[0], mc.e_tot, 8)
        for opt in range (2):
            with self.subTest (opt=opt):
                lasdm1s, lasdm2s = root_make_rdm12s (las, las.ci, si, state=0, opt=opt)
                lasdm1 = lasdm1s.sum (0)
                lasdm2 = lasdm2s.sum ((0,3))
                with self.subTest ("casdm1"):
                    self.assertAlmostEqual (lib.fp (lasdm1), lib.fp (casdm1), 8)
                with self.subTest ("casdm2"):
                    self.assertAlmostEqual (lib.fp (lasdm2), lib.fp (casdm2), 8)
                if opt<2:
                    stdm1s = make_stdm12s (las, opt=opt)[0][9:13,:,:,:,9:13] # second rootspace
                    with self.subTest("state indexing"):
                        # column-major ordering for state excitation quantum numbers:
                        # earlier fragments advance faster than later fragments
                        self.assertAlmostEqual (lib.fp (stdm1s[0,:,:2,:2,0]),
                                                lib.fp (stdm1s[2,:,:2,:2,2]))
                        self.assertAlmostEqual (lib.fp (stdm1s[0,:,2:,2:,0]),
                                                lib.fp (stdm1s[1,:,2:,2:,1]))

    @unittest.skip("debugging")
    def test_si_trans_rdm12s (self):
        las, e_roots, si_ket = lsi._las, lsi.e_roots, lsi.si
        si_bra = np.roll (si_ket, 1, axis=1)
        stdm1s, stdm2s = make_stdm12s (las, opt=0)
        rdm1s_ref = lib.einsum ('ir,jr,isabj->rsab', si_bra.conj (), si_ket, stdm1s)
        rdm2s_ref = lib.einsum ('ir,jr,isabtcdj->rsabtcd', si_bra.conj (), si_ket, stdm2s)
        for opt in range (2):
            with self.subTest (opt=opt):
                lasdm1s, lasdm2s = roots_trans_rdm12s (las, las.ci, si_bra, si_ket, opt=opt)
                with self.subTest ("lasdm1s"):
                    self.assertAlmostEqual (lib.fp (lasdm1s), lib.fp (rdm1s_ref), 8)
                with self.subTest ("lasdm2s"):
                    self.assertAlmostEqual (lib.fp (lasdm2s), lib.fp (rdm2s_ref), 8)

    def test_davidson (self):
        lsi1 = LASSI (lsi._las).run (davidson_only=True)
        self.assertAlmostEqual (lsi1.e_roots[0], lsi.e_roots[0], 8)
        ovlp = np.dot (lsi1.si[:,0], lsi.si[:,0].conj ()) ** 2.0
        self.assertAlmostEqual (ovlp, 1.0, 4)
        lsi1.eig ()

    @unittest.skip("debugging")
    def test_lassirq (self):
        lsi1 = LASSIrq (las, 2, 3).run ()
        self.assertAlmostEqual (lsi1.e_roots[0], mc.e_tot, 8)

    @unittest.skip("debugging")
    def test_lassirqct (self):
        lsi1 = LASSIrqCT (las, 2, 3).run ()
        self.assertAlmostEqual (lsi1.e_roots[0], -4.2879945248402445, 8)

    @unittest.skip("debugging")
    def test_contract_hlas_ci (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        case_contract_hlas_ci (self, las, h0, h1, h2, las.ci, lsi.get_nelec_frs ())

    @unittest.skip("debugging")
    def test_contract_op_si (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        case_contract_op_si (self, las, h1, h2, las.ci, lsi.get_nelec_frs ())

    @unittest.skip("debugging")
    def test_energy_tot (self):
        las1 = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1)).run () # different MOs
        # You need to set up at least the rootspaces because otherwise LASSI doesn't
        # have any way of knowing how many electrons are in each CI vector I guess
        for i in range (2): las1 = all_single_excitations (las1)
        las1.conv_tol_grad = las.conv_tol_self = 9e99
        las1.lasci ()
        lsi1 = LASSI (las1).run ()
        e_tot = lsi1.energy_tot (mo_coeff=lsi.mo_coeff, ci=lsi.ci, si=lsi.si)
        self.assertAlmostEqual (lib.fp (e_tot), lib.fp (lsi.e_roots), 9)

    @unittest.skip("debugging")
    def test_lassis (self):
        for opt in (0,1):
            for dson in (False,True):
                with self.subTest (opt=opt, davidson_only=dson):
                    lsis.run (opt=opt, davidson_only=dson, nroots_si=20)
                    e_upper = las.e_states[0]
                    e_lower = lsi.e_roots[0]
                    self.assertLessEqual (e_lower, lsis.e_roots[0])
                    self.assertLessEqual (lsis.e_roots[0], e_upper)
                    self.assertEqual (len (lsis.e_roots), 20)
                    # Reference depends on rng seed obviously b/c this is not casci limit
                    self.assertAlmostEqual (lsis.e_roots[0], -4.134472877702426, 8)
                    case_lassis_fbf_2_model_state (self, lsis)
                    case_lassis_fbfdm (self, lsis)

    @unittest.skip("debugging")
    def test_lassis_energy_tot (self):
        lsis1 = lassis.LASSIS (las)
        e_tot = lsis1.energy_tot (
            mo_coeff=lsis.mo_coeff,
            ci_ref=lsis.get_ci_ref (),
            ci_sf=lsis.ci_spin_flips,
            ci_ch=lsis.ci_charge_hops,
            si=lsis.si
        )
        self.assertAlmostEqual (lib.fp (e_tot), lib.fp (lsis.e_roots), 9)

    @unittest.skip("debugging")
    def test_lassis_ugg (self):
        case_lassis_ugg (self, lsis)

    @unittest.skip("debugging")
    def test_lassis_grads (self):
        for lsis.opt in range (2):
            case_lassis_grads (self, lsis)

    @unittest.skip("debugging")
    def test_lassis_hessian (self):
        for lsis.opt in range (2):
            case_lassis_hessian (self, lsis)

    @unittest.skip("debugging")
    def test_fdm1 (self):
        make_fdm1 = get_fdm1_maker (lsi, lsi.ci, lsi.get_nelec_frs (), lsi.si)
        for iroot in range (lsi.nroots):
            for ifrag in range (lsi.nfrags):
                with self.subTest (iroot=iroot, ifrag=ifrag):
                    fdm1 = make_fdm1 (iroot, ifrag)
                    sdm1 = make_sdm1 (lsi, iroot, ifrag)
                    self.assertAlmostEqual (lib.fp (fdm1), lib.fp (sdm1), 7)

if __name__ == "__main__":
    print("Full Tests for LASSI of random 2,2 system")
    unittest.main()

