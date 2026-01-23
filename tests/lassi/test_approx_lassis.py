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
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, scf
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, LASSIS

def setUpModule ():
    global las, mask 
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g')
    mol.verbose = 0 #lib.logger.DEBUG
    mol.output = '/dev/null' #'test_c2h4n4.log'
    mol.spin = 8
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
    las.kernel (mo_coeff)
    mask = np.ones ((3,3), dtype=bool)
    mask[0,2] = mask[2,0] = False

def tearDownModule():
    global las, mask
    mol = las.mol
    mol.stdout.close ()
    del las, mask

class KnownValues(unittest.TestCase):

    def test_full_lassis (self):
        lsi = LASSIS (las).run ()
        self.assertTrue (lsi.converged)
        self.assertEqual (lsi.si.shape[1], 321)
        self.assertAlmostEqual (lsi.e_roots[0], -295.52016414968716, 6)

    def test_fewer_charge_hops (self):
        lsi = LASSIS (las).run (mask_charge_hops=mask)
        self.assertTrue (lsi.converged)
        self.assertEqual (lsi.si.shape[1], 213)
        self.assertAlmostEqual (lsi.e_roots[0], -295.4857294756858, 6)

    def test_no_spin_flips (self):
        lsi = LASSIS (las).run (nspin=0)
        self.assertTrue (lsi.converged)
        self.assertEqual (lsi.si.shape[1], 25)
        self.assertAlmostEqual (lsi.e_roots[0], -295.47881153342126, 6)

    def test_ultra_small (self):
        lsi = LASSIS (las).run (mask_charge_hops=mask, nspin=0)
        self.assertTrue (lsi.converged)
        self.assertEqual (lsi.si.shape[1], 9)
        self.assertAlmostEqual (lsi.e_roots[0], -295.47263837707146, 6)

if __name__ == "__main__":
    print("Full Tests for approximate LASSIS")
    unittest.main()

