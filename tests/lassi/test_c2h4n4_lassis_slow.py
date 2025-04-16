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
from mrh.my_pyscf.lassi import LASSI, LASSIS
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_lassis_fbf_2_model_state
from mrh.tests.lassi.addons import case_lassis_fbfdm, case_contract_op_si
from mrh.tests.lassi.addons import case_lassis_grads, case_lassis_hessian
topdir = os.path.abspath (os.path.join (__file__, '..'))

def setUpModule ():
    global las, lsi, rdm1s, rdm2s
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = 0 #lib.logger.DEBUG
    mol.output = '/dev/null' #'test_c2h4n4.log'
    mol.spin = 8
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (5,5), ((3,2),(2,3)), spin_sub=(2,2))
    mo_coeff = las.localize_init_guess ((list (range (5)), list (range (5,10))))
    las.kernel (mo_coeff)
    lsi = LASSIS (las).run ()

def tearDownModule():
    global las, lsi
    mol = lsi._las.mol
    mol.stdout.close ()
    del las, lsi

class KnownValues(unittest.TestCase):

    def test_energy (self):
        for dson in (False, True):
            for opt in (0,1):
                with self.subTest (opt=opt, davidson_only=dson):
                    lsi.run (opt=opt, davidson_only=dson)
                    self.assertTrue (lsi.converged)
                    self.assertAlmostEqual (lsi.e_roots[0], -295.52185731568903, 7)

    def test_fbf_2_model_state (self):
        for lsi.opt in (0,1):
            with self.subTest (opt=lsi.opt):
                case_lassis_fbf_2_model_state (self, lsi)

    def test_fbfdm (self):
        for lsi.opt in (0,1):
            with self.subTest (opt=lsi.opt):
                case_lassis_fbfdm (self, lsi)

    def test_as_scanner (self):
        for dson in (False, True):
            with self.subTest (davidson_only=dson):
                lsi_scanner = lsi.as_scanner ()
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
                self.assertTrue (lsi2.converged)
                self.assertAlmostEqual (lsi_scanner.e_roots[0], lsi2.e_roots[0], 5)

    #def test_grads (self):
    #    for lsi.opt in (0,1):
    #        with self.subTest (opt=lsi.opt):
    #            case_lassis_grads (self, lsi)    

    #def test_hessian (self):
    #    for lsi.opt in (0,1):
    #        with self.subTest (opt=lsi.opt):
    #            case_lassis_hessian (self, lsi)    

if __name__ == "__main__":
    print("Full Tests for LASSIS of c2h4n4 molecule")
    unittest.main()

