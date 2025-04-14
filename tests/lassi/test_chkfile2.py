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
import tempfile
import numpy as np
from pyscf.tools import molden
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi

def setUpModule():
    global mf, lsi, lsi2
    xyz='''Li 0 0 0,
           H 2 0 0,
           Li 10 0 0,
           H 12 0 0'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=None, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    from pyscf.tools import molden
    mc = mcscf.CASSCF (mf, 4, 4).run ()
    mc.analyze ()
    
    las = LASSCF (mf, (2,2), (2,2))
    mo = las.localize_init_guess (([0,1],[2,3]), mc.mo_coeff, freeze_cas_spaces=True)
    las.kernel (mo)
    lsi = lassi.LASSIrq (las, r=2, q=2).run ()
    with tempfile.NamedTemporaryFile() as chkfile:
        lsi.dump_chk (chkfile=chkfile.name)
        lsi2 = lassi.LASSIrq (las, r=2, q=2)
        lsi2.load_chk_(chkfile=chkfile.name)

def tearDownModule():
    global mf, lsi, lsi2
    mf.mol.stdout.close ()
    del mf, lsi, lsi2

class KnownValues(unittest.TestCase):
    def test_config (self):
        self.assertEqual (lsi.ncore, lsi2.ncore)
        self.assertEqual (lsi.nfrags, lsi2.nfrags)
        self.assertEqual (lsi.nroots, lsi2.nroots)
        self.assertEqual (lsi.break_symmetry, lsi2.break_symmetry)
        self.assertEqual (lsi.soc, lsi2.soc)
        self.assertEqual (lsi.opt, lsi2.opt)
        self.assertListEqual (list(lsi.ncas_sub), list(lsi2.ncas_sub))
        self.assertListEqual (lsi.nelecas_sub.tolist(), lsi2.nelecas_sub.tolist ())
        self.assertListEqual (list(lsi.weights), list(lsi2.weights))

    def test_results (self):
        self.assertListEqual (list(lsi.e_states), list(lsi2.e_states))
        self.assertListEqual (list(lsi.e_roots), list(lsi2.e_roots))
        self.assertListEqual (list(lsi.s2), list(lsi2.s2))
        for ne, ne2 in zip (lsi.nelec, lsi2.nelec):
            self.assertListEqual (list(ne), list(ne2))
        self.assertListEqual (list(lsi.wfnsym), list(lsi2.wfnsym))
        for r, r2 in zip (lsi.rootsym, lsi2.rootsym):
            self.assertListEqual (list(r), list(r2))
        self.assertAlmostEqual (lib.fp (lsi.si), lib.fp (lsi2.si), 9)
        for i in range (2):
            for j in range (lsi.nroots):
                self.assertAlmostEqual (lib.fp (lsi.ci[i][j]), lib.fp (lsi2.ci[i][j]), 9)

if __name__ == "__main__":
    print("Full Tests for LASSCF chkfile")
    unittest.main()

