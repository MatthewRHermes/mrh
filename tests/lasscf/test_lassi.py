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
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_testing import LASSCF

dr_nn = 2.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry='Cs')
mol.verbose = lib.logger.DEBUG 
mol.output = 'test_lassi.log'
mol.spin = 0 
mol.symmetry = 'Cs'
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
mo_loc = las.localize_init_guess ((list(range(3)),list(range(7,10))), mo_coeff=mf.mo_coeff)
las.state_average_(weights=[1.0/7.0,]*7,
    spins=[[0,0],[0,0],[2,-2],[-2,2],[0,0],[0,0],[2,2]],
    smults=[[1,1],[3,3],[3,3],[3,3],[1,1],[1,1],[3,3]],
    wfnsyms=[['A\'','A\''],]*4+[['A"','A\''],['A\'','A"'],['A\'','A\'']])
mo_loc = np.loadtxt ('test_lassi_mo.dat')
las.frozen = list (range (mo_loc.shape[-1]))
las.set (conv_tol_grad=1e-6).kernel (mo_loc)
e_roots, si = las.lassi ()

def tearDownModule():
    global mol, mf, las
    mol.stdout.close ()
    del mol, mf, las

class KnownValues(unittest.TestCase):
    def test_evals (self):
        self.assertAlmostEqual (lib.fp (e_roots), -499.9894024534505, 6)

    def test_si (self):
        # Arbitrary signage in both the SI and CI vector, sadly
        dms = [np.dot (si[:,i:i+1], si[:,i:i+1].conj ().T) for i in range (7)]
        self.assertAlmostEqual (lib.fp (np.abs (dms)), 3.645523608610042, 7)

    def test_nelec (self):
        self.assertEqual (si.nelec[0], (6,2))
        for ne in si.nelec[1:]:
            self.assertEqual (ne, (4,4))

    def test_s2 (self):
        s2_array = np.zeros (7)
        s2_array[2] = 6
        s2_array[3] = 2
        self.assertAlmostEqual (lib.fp (si.s2), lib.fp (s2_array), 3)

    def test_wfnsym (self):
        self.assertEqual (si.wfnsym, [0,]*5 + [1,]*2)

if __name__ == "__main__":
    print("Full Tests for SA-LASSI")
    unittest.main()

