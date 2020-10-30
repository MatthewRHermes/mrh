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
from mrh.my_pyscf.mcscf.lasscf_testing import LASSCF
from mrh.my_pyscf.mcscf.lassi import roots_make_rdm12s, make_stdm12s, ham_2q
from mrh.my_pyscf.mcscf.lasci import get_init_guess_ci

# Build crazy state list
states  = {'charges': [[0,0,0],],
           'spins':   [[0,0,0],],
           'smults':  [[1,1,1],],
           'wfnsyms': [[0,0,0],]}
states1 = {'charges': [[-1,1,0],[-1,1,0],[1,-1,0],[1,-1,0],[0,1,-1],[0,1,-1],[0,-1,1],[0,-1,1]],
           'spins':   [[-1,1,0],[1,-1,0],[-1,1,0],[1,-1,0],[0,1,-1],[0,-1,1],[0,1,-1],[0,-1,1]],
           'smults':  [[2,2,1], [2,2,1], [2,2,1], [2,2,1], [1,2,2], [1,2,2], [1,2,2], [1,2,2]],
           'wfnsyms': [[1,1,0], [1,1,0], [1,1,0], [1,1,0], [0,1,1], [0,1,1], [0,1,1], [0,1,1]]}
states2 = {'charges': [[0,0,0],]*6,
           'spins':   [[2,-2,0],[0,0,0],[-2,2,0],[0,2,-2],[0,0,0],[0,-2,2]],
           'smults':  [[3,3,1], [3,3,1],[3,3,1], [1,3,3], [1,3,3],[1,3,3]],
           'wfnsyms': [[0,0,0],]*6}
states3 = {'charges': [[-1,2,-1],[-1,2,-1],[1,-2,1],[1,-2,1],[-1,0,1],[-1,0,1],[1,0,-1],[1,0,-1]],
           'spins':   [[1,0,-1], [-1,0,1], [1,0,-1],[-1,0,1],[1,0,-1],[-1,0,1],[1,0,-1],[-1,0,1]],
           'smults':  [[2,1,2],  [2,1,2],  [2,1,2], [2,1,2], [2,1,2], [2,1,2], [2,1,2], [2,1,2]],
           'wfnsyms': [[1,0,1],]*8}
states4 = {'charges': [[0,0,0],]*10,
           'spins':   [[-2,0,2],[0,0,0],[2,0,-2],[-2,0,2],[0,0,0],[2,0,-2],[2,-2,0],[-2,2,0],[0,2,-2],[0,-2,2]],
           'smults':  [[3,1,3], [3,1,3],[3,1,3], [3,3,3], [3,3,3],[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
           'wfnsyms': [[0,0,0],]*10}
states5 = {'charges': [[-1,1,0],[-1,1,0], [-1,1,0],[-1,1,0],[1,-1,0],[1,-1,0], [1,-1,0],[1,-1,0]],
         'spins':   [[1,1,-2],[-1,-1,2],[1,-1,0],[-1,1,0],[1,1,-2],[-1,-1,2],[1,-1,0],[-1,1,0]],
         'smults':  [[2,2,3], [2,2,3],  [2,2,3], [2,2,3], [2,2,3], [2,2,3],  [2,2,3], [2,2,3]],
         'wfnsyms': [[1,1,0],]*8}
states6 = deepcopy (states5)
states7 = deepcopy (states5)
for field in ('charges', 'spins', 'smults', 'wfnsyms'):
    states6[field] = [[row[1], row[2], row[0]] for row in states5[field]]
    states7[field] = [[row[2], row[0], row[1]] for row in states5[field]]
for d in [states1, states2, states3, states4, states5, states6, states7]:
    for field in ('charges', 'spins', 'smults', 'wfnsyms'):
        states[field] = states[field] + d[field]
weights = [1.0,] + [0.0,]*56
nroots = 57
# End building crazy state list

dr_nn = 2.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry='Cs')
mol.verbose = lib.logger.INFO 
mol.output = 'test_lassi_op57.log'
mol.spin = 0 
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,2,4), (4,2,4))
las.state_average_(weights=weights, **states)
las.mo_coeff = las.localize_init_guess ((list (range (3)),
    list (range (3,7)), list (range (7,10))), mf.mo_coeff)
las.ci = get_init_guess_ci (las, las.mo_coeff, las.get_h2eff (las.mo_coeff))
las.e_states = np.zeros (57)
np.random.seed (1)
for c in las.ci:
    for iroot in range (len (c)):
        c[iroot] = np.random.rand (*c[iroot].shape)
        c[iroot] /= linalg.norm (c[iroot])

def tearDownModule():
    global mol, mf, las
    mol.stdout.close ()
    del mol, mf, las

class KnownValues(unittest.TestCase):
    def test_stdm12s (self):
        d12_o0 = make_stdm12s (las, o=0)
        d12_o1 = make_stdm12s (las, o=1)
        for r in range (2):
            for i, j in product (range (nroots), repeat=2):
                with self.subTest (rank=r+1, bra=i, ket=j):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i,...,j]),
                        lib.fp (d12_o1[r][i,...,j]), 9)

if __name__ == "__main__":
    print("Full Tests for LASSI matrix elements of 57-state manifold")
    unittest.main()

