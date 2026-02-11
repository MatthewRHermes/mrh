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
import os
import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import roots_make_rdm12s, root_make_rdm12s, make_stdm12s, ham_2q
from mrh.my_pyscf.lassi import LASSI, LASSIS, op_o0, op_o1
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_lassis_fbf_2_model_state
from mrh.tests.lassi.addons import case_lassis_fbfdm, case_contract_op_si
from mrh.tests.lassi.addons import case_lassis_grads, case_lassis_hessian, case_lassis_ugg
from mrh.tests.lassi.addons import case_matrix_o0_o1
topdir = os.path.abspath (os.path.join (__file__, '..'))
op = (op_o0, op_o1)

def setUpModule ():
    global las, lsis, ham, s2, ovlp
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = 0 #lib.logger.DEBUG
    mol.output = '/dev/null' #'test_c2h4n4.log'
    mol.spin = 8
    mol.build ()
    mf = scf.RHF (mol).run ()
    #las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    #mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
    las = LASSCF (mf, (5,5), ((3,2),(2,3)), spin_sub=(2,2))
    mo_coeff = las.localize_init_guess ((list (range (5)), list (range (5,10))))
    las.kernel (mo_coeff)
    lsi_o1 = LASSIS (las).run (nroots_si=3)
    assert (lsi_o1.opt==1)
    assert (not lsi_o1.sisolver.davidson_only)
    lsi_o0 = LASSIS (las, opt=0).run ()
    assert (lsi_o0.opt==0)
    assert (not lsi_o0.sisolver.davidson_only)
    lsis = [lsi_o0, lsi_o1]
    h0, h1, h2 = lsis[0].ham_2q ()
    nelec_frs = lsis[0].get_nelec_frs ()
    smult_fr = lsis[0].get_smult_fr ()
    ham_o0, s2_o0, ovlp_o0 = op[0].ham (lsis[1], h1, h2, lsis[1].ci, nelec_frs)[:3]
    ham_o1, s2_o1, ovlp_o1 = op[1].ham (lsis[1], h1, h2, lsis[1].ci, nelec_frs,
                                        smult_fr=smult_fr)[:3]
    ham = [ham_o0, ham_o1]
    s2 = [s2_o0, s2_o1]
    ovlp = [ovlp_o0, ovlp_o1]

def tearDownModule():
    global las, lsis, ham, s2, ovlp
    mol = lsis[0]._las.mol
    mol.stdout.close ()
    del las, lsis, ham, s2, ovlp

class KnownValues(unittest.TestCase):

    #@unittest.skip("debugging")
    def test_lassis_kernel_noniterative (self):
        for lsi in lsis:
            with self.subTest (opt=lsi.opt):
                self.assertTrue (lsi.converged)
                self.assertAlmostEqual (lsi.e_roots[0], -295.52185731568903, 7)

    #@unittest.skip("debugging")
    def test_lassis_kernel_davidson (self):
        for lsi in lsis:
            with self.subTest (opt=lsi.opt):
                lsi1 = lsi.copy ()
                e_roots, si = lsi1.eig (davidson_only=True)
                self.assertTrue (lsi1.converged)
                self.assertAlmostEqual (e_roots[0], -295.52185731568903, 7)

    def test_lassis_kernel_lsf (self):
        for lsi in lsis:
            with self.subTest (opt=lsi.opt):
                lsi1 = lsi.copy ()
                e_roots, si = lsi1.eig (davidson_only=True, smult_si=1)
                self.assertTrue (lsi1.converged)
                self.assertAlmostEqual (e_roots[0], -295.52185731568903, 7)
                self.assertAlmostEqual (lsi1.s2[0], 0.0, 7)

    def test_o1_ham (self):
        case_matrix_o0_o1 (self, ham[0], ham[1],
                           lsis[0].get_nelec_frs (),
                           lsis[0].get_lroots (),
                           lsis[0].get_smult_fr ())

    def test_o1_s2 (self):
        case_matrix_o0_o1 (self, s2[0], s2[1],
                           lsis[0].get_nelec_frs (),
                           lsis[0].get_lroots (),
                           lsis[0].get_smult_fr ())

    def test_o1_ovlp (self):
        case_matrix_o0_o1 (self, ovlp[0], ovlp[1],
                           lsis[0].get_nelec_frs (),
                           lsis[0].get_lroots (),
                           lsis[0].get_smult_fr ())

    def test_o1_contract_op_si (self):
        lsi = lsis[1]
        las = lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        case_contract_op_si (self, las, h1, h2, lsi.ci, lsi.get_nelec_frs (),
                             smult_fr=lsi.get_smult_fr (),
                             disc_fr=lsi.get_disc_fr ())

    #@unittest.skip("debugging")
    def test_fbf_2_model_state (self):
        for lsi in lsis:
            with self.subTest (opt=lsi.opt):
                case_lassis_fbf_2_model_state (self, lsi)

    #@unittest.skip("debugging")
    def test_fbfdm (self):
        for lsi in lsis:
            with self.subTest (opt=lsi.opt):
                case_lassis_fbfdm (self, lsi)

    #@unittest.skip("debugging")
    def test_as_scanner (self):
        for dson in (False, True):
            with self.subTest (davidson_only=dson):
                # TODO: copy method for LASSIS that does all this
                lsi_scanner = lsis[1].copy ().as_scanner ()
                mol2 = struct (1.9, 1.9, '6-31g', symmetry=False)
                mol2.verbose = 0
                mol2.output = '/dev/null'
                mol2.build ()
                lsi_scanner (mol2)
                self.assertTrue (lsi_scanner.converged)
                mf2 = scf.RHF (mol2).run ()
                las2 = LASSCF (mf2, (5,5), ((3,2),(2,3)), spin_sub=(2,2))
                las2.mo_coeff = lsi_scanner.mo_coeff
                las2.lasci ()
                lsi2 = LASSIS (las2).run (davidson_only=dson)
                assert (lsi2.opt==1)
                self.assertTrue (lsi2.converged)
                self.assertAlmostEqual (lsi_scanner.e_roots[0], lsi2.e_roots[0], 5)

    #@unittest.skip("debugging")
    def test_lassis_ugg (self):
        case_lassis_ugg (self, lsis[1])

    #@unittest.skip("debugging")
    def test_grads (self):
        case_lassis_grads (self, lsis[1])

    #@unittest.skip("debugging")
    def test_hessian (self):
        case_lassis_hessian (self, lsis[1])

if __name__ == "__main__":
    print("Full Tests for LASSIS of c2h4n4 molecule")
    unittest.main()

