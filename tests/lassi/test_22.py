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
from pyscf import lib, gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, make_stdm12s
from mrh.my_pyscf.lassi.states import all_single_excitations
from mrh.my_pyscf.mcscf.lasci import get_space_info

def setUpModule ():
    global mol, mf
    xyz='''H 0 0 0
    H 1 0 0
    H 3 0 0
    H 4 0 0'''
    mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()

    # Random Hamiltonian
    rng = np.random.default_rng ()
    mf._eri = rng.random (mf._eri.shape)
    hcore = rng.random ((4,4))
    mf.get_hcore = lambda *args: hcore

def tearDownModule():
    global mol, mf
    mol.stdout.close ()
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_casci_limit (self):
        # CASCI limit
        mc = mcscf.CASCI (mf, 4, 4).run ()
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)

        # LASSI in the CASCI limit
        las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
        for i in range (2): las = all_single_excitations (las)
        charges, spins, smults, wfnsyms = get_space_info (las)
        lroots = 4 - smults
        idx = (charges!=0) & (lroots==3)
        lroots[idx] = 1
        las.conv_tol_grad = las.conv_tol_self = 9e99
        las.lasci (lroots=lroots.T)
        e_roots, si = las.lassi (opt=0)
        with self.subTest ("total energy"):
            self.assertAlmostEqual (e_roots[0], mc.e_tot, 8)
        lasdm1s, lasdm2s = root_make_rdm12s (las, las.ci, si, state=0, opt=0)
        lasdm1 = lasdm1s.sum (0)
        lasdm2 = lasdm2s.sum ((0,3))
        with self.subTest ("casdm1"):
            self.assertAlmostEqual (lib.fp (lasdm1), lib.fp (casdm1), 8)
        with self.subTest ("casdm2"):
            self.assertAlmostEqual (lib.fp (lasdm2), lib.fp (casdm2), 8)
        stdm1s = make_stdm12s (las, opt=0)[0][9:13,:,:,:,9:13] # second rootspace
        with self.subTest("state indexing"):
            # column-major ordering for state excitation quantum numbers:
            # earlier fragments advance faster than later fragments
            self.assertAlmostEqual (lib.fp (stdm1s[0,:,:2,:2,0]),
                                    lib.fp (stdm1s[2,:,:2,:2,2]))
            self.assertAlmostEqual (lib.fp (stdm1s[0,:,2:,2:,0]),
                                    lib.fp (stdm1s[1,:,2:,2:,1]))

if __name__ == "__main__":
    print("Full Tests for LASSI of random 2,2 system")
    unittest.main()

