#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
#
# Author: Bhavnesh Jangid <email: jangidbhavnesh@uchicago.edu>
#

import unittest
from pyscf import gto, scf
from mrh.my_pyscf.dkh import dkh

class KnownValues(unittest.TestCase):
    
    def test_DKH2(self):
        mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=0)
        mol.build()
        mfdkh = scf.RHF(mol)
        mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=2,c=137.0359895) # Orca's Speed of Light
        mfdkh.kernel()
        self.assertAlmostEqual(mfdkh.e_tot, -128.62532864034782, 7)
    
    def test_DKH3(self):
        mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=0)
        mol.build()
        mfdkh = scf.RHF(mol)
        mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=3,c=137.0359895) # Orca's Speed of Light
        mfdkh.kernel()
        self.assertAlmostEqual(mfdkh.e_tot, -128.62538501869074, 7)

    def test_DKH4(self):
        mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=0)
        mol.build()
        mfdkh = scf.RHF(mol)
        mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=4,c=137.0359895) # Orca's Speed of Light
        mfdkh.kernel()
        self.assertAlmostEqual(mfdkh.e_tot, -128.62538062385389, 7)

if __name__ == "__main__":
    print("Test for DKH Scalar Relativisitic Effects")
    unittest.main()


