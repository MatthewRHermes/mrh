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
import me2n2_driver as driver

r_nn = 3.0
mol = struct (3.0, '6-31g')
mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 4, 4).run ()
mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()
mc_df = mcscf.CASSCF (mf_df, 4, 4).run ()

def tearDownModule():
    global mol, mf, mf_df, mc, mc_df
    mol.stdout.close()
    del mol, mf, mf_df, mc, mc_df


class KnownValues(unittest.TestCase):
    def test_lasscf (self):
        elas = driver.run (3.0, do_df=False)
        self.assertAlmostEqual (elas, mc.e_tot, 7)

    def test_lasscf_df (self):
        elas = driver.run (3.0, do_df=True)
        self.assertAlmostEqual (elas, mc_df.e_tot, 7)

if __name__ == "__main__":
    print("Full Tests for LASSCF me2n2")
    unittest.main()

