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
mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
mol.verbose = 0 
mol.output = '/dev/null'
mol.spin = 0 
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3))
las.max_cycle_macro = 1
las.kernel ()
las.mo_coeff = np.loadtxt ('test_lasci_mo.dat')
las.ci = [[np.loadtxt ('test_lasci_ci0.dat')], [-np.loadtxt ('test_lasci_ci1.dat').T]]
ugg = las.get_ugg (las, las.mo_coeff, las.ci)
h_op = las.get_hop (las, ugg)
np.random.seed (0)
x = np.random.rand (ugg.nvar_tot)

def tearDownModule():
    global mol, mf, las, ugg, h_op, x
    mol.stdout.close ()
    del mol, mf, las, ugg, h_op, x


class KnownValues(unittest.TestCase):
    def test_grad (self):
        gorb0, gci0, gx0 = las.get_grad (ugg=ugg)
        grad0 = np.append (gorb0, gci0)
        grad1 = h_op.get_grad ()
        gx1 = h_op.get_gx ()
        self.assertAlmostEqual (lib.fp (grad0), -0.1547273632764783, 9)
        self.assertAlmostEqual (lib.fp (grad1), -0.1547273632764783, 9)
        self.assertAlmostEqual (lib.fp (gx0), -0.0005604501808183955, 9)
        self.assertAlmostEqual (lib.fp (gx1), -0.0005604501808183955, 9)

    def test_hessian (self):
        hx = h_op._matvec (x)
        self.assertAlmostEqual (lib.fp (hx), 178.92725229582146, 9)

    def test_hc2 (self):
        xp = x.copy ()
        xp[:-16] = 0.0
        hx = h_op._matvec (xp)[-16:]
        self.assertAlmostEqual (lib.fp (hx), -0.5385952489125434, 9)

    def test_hcc (self):
        xp = x.copy ()
        xp[:-16] = 0.0
        hx = h_op._matvec (xp)[-32:-16]
        self.assertAlmostEqual (lib.fp (hx), -0.001474000383931805, 9)

    def test_hco (self):
        xp = x.copy ()
        xp[-32:] = 0.0
        hx = h_op._matvec (xp)[-32:]
        self.assertAlmostEqual (lib.fp (hx), 0.2698490298969052, 9)

    def test_hoc (self):
        xp = x.copy ()
        xp[:-32] = 0.0
        hx = h_op._matvec (xp)[:-32]
        self.assertAlmostEqual (lib.fp (hx), -0.029477903804816963, 9)

    def test_hoo (self):
        xp = x.copy ()
        xp[-32:] = 0.0
        hx = h_op._matvec (xp)[:-32]
        self.assertAlmostEqual (lib.fp (hx), 178.29669908809024, 9)

    def test_prec (self):
        M_op = h_op.get_prec ()
        Mx = M_op._matvec (x)
        self.assertAlmostEqual (lib.fp (Mx), 8358.536413578968, 7)


if __name__ == "__main__":
    print("Full Tests for LASSCF Newton-CG module functions")
    unittest.main()

