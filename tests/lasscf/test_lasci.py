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
from copy import deepcopy
from itertools import product
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

xyz = '''6        2.215130000      3.670330000      0.000000000
1        3.206320000      3.233120000      0.000000000
1        2.161870000      4.749620000      0.000000000
6        1.117440000      2.907720000      0.000000000
1        0.141960000      3.387820000      0.000000000
1       -0.964240000      1.208850000      0.000000000
6        1.117440000      1.475850000      0.000000000
1        2.087280000      0.983190000      0.000000000
6        0.003700000      0.711910000      0.000000000
6       -0.003700000     -0.711910000      0.000000000
6       -1.117440000     -1.475850000      0.000000000
1        0.964240000     -1.208850000      0.000000000
1       -2.087280000     -0.983190000      0.000000000
6       -1.117440000     -2.907720000      0.000000000
6       -2.215130000     -3.670330000      0.000000000
1       -0.141960000     -3.387820000      0.000000000
1       -2.161870000     -4.749620000      0.000000000
1       -3.206320000     -3.233120000      0.000000000'''
mol = gto.M (atom = xyz, basis='STO-3G', symmetry=False, verbose=0, output='/dev/null')
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (2,2,2,2),((1,1),(1,1),(1,1),(1,1)))
a = list (range (18))
frags = [a[:5], a[5:9], a[9:13], a[13:18]]
mo = las.localize_init_guess (frags, mf.mo_coeff)
del a, frags
#las.kernel (mo)

# State list contains a couple of different 4-frag interactions
states  = {'charges': [[0,0,0,0],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1]],
           'spins':   [[0,0,0,0],[1,1,-1,-1],[0,1,-1,0], [1,0,0,-1],[-1,-1,1,1],[0,-1,1,0], [-1,0,0,1]],
           'smults':  [[1,1,1,1],[2,2,2,2],  [1,2,2,1],  [2,1,1,2], [2,2,2,2],  [1,2,2,1],  [2,1,1,2]],
           'wfnsyms': [[0,0,0,0],]*7}
weights = [1.0,] + [0.0,]*6
las_ref = [None]

def _check_():
    if not las.converged: las.kernel (mo)
    if las_ref[0] is None:
        las_ref[0] = las.state_average (weights=weights, **states)
        las_ref[0].frozen = range (mo.shape[1])
        las_ref[0].kernel ()

def tearDownModule():
    global mol, mf, las, las_ref, states, weights, mo, _check_
    mol.stdout.close ()
    del mol, mf, las, las_ref, states, weights, mo, _check_

class KnownValues(unittest.TestCase):
    def test_sanity (self):
        _check_()
        las_test = las_ref[0].state_average (weights=weights, **states)
        las_test.lasci ()
        self.assertAlmostEqual (lib.fp (las_test.e_states), lib.fp (las_ref[0].e_states), 5)

    def test_convergence (self):
        _check_()
        las_test = las.state_average (weights=weights, **states)
        las_test.lasci ()
        self.assertAlmostEqual (lib.fp (las_test.e_states), lib.fp (las_ref[0].e_states), 5)



if __name__ == "__main__":
    print("Full Tests for LASCI calculation")
    unittest.main()

