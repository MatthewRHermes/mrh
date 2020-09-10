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
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from me2n2_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_testing import LASSCF

r_nn = 3.0
mol = struct (3.0, '6-31g')
mol.output = 'test_me2n2.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 4, 4).run ()
mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()
mc_df = mcscf.CASSCF (mf_df, 4, 4).run ()

mol_hs = mol.copy ()
mol_hs.spin = 4
mol_hs.build ()
mf_hs = scf.RHF (mol_hs).run ()
mf_hs_df = mf_hs.density_fit (auxbasis = df.aug_etb (mol_hs)).run ()

def tearDownModule():
    global mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df
    mol.stdout.close ()
    mol_hs.stdout.close ()
    del mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df


class KnownValues(unittest.TestCase):
    def test_lasscf (self):
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5).run ()
        self.assertAlmostEqual (las.e_tot, mc.e_tot, 8)

    def test_lasscf_df (self):
        las = LASSCF (mf_df, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-5).run ()
        self.assertAlmostEqual (las.e_tot, mc_df.e_tot, 8)

    def test_lasscf_hs (self):
        las = LASSCF (mf_hs, (4,), ((4,0),), spin_sub=(5,)).set (conv_tol_grad=1e-5).run ()
        self.assertAlmostEqual (las.e_tot, mf_hs.e_tot, 8)

    def test_lasscf_hs_df (self):
        las = LASSCF (mf_hs_df, (4,), ((4,0),), spin_sub=(5,)).set (conv_tol_grad=1e-5).run ()
        self.assertAlmostEqual (las.e_tot, mf_hs_df.e_tot, 8)

if __name__ == "__main__":
    print("Full Tests for LASSCF me2n2")
    unittest.main()

