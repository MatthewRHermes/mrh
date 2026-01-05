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
from pyscf import lib, gto, scf, dft, fci, mcscf
from pyscf.tools import molden
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import root_trans_rdm12s
from mrh.my_pyscf.lassi.lassi import make_stdm12s, ham_2q, las_symm_tuple
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1
from mrh.my_pyscf.lassi import LASSIS
from mrh.my_pyscf.lassi.op_o1.utilities import lst_hopping_index, get_scallowed_interactions
from mrh.my_pyscf.lassi.op_o1 import get_fdm1_maker
from mrh.my_pyscf.lassi.sitools import make_sdm1
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_lassis_fbf_2_model_state
from mrh.tests.lassi.addons import case_lassis_fbfdm, case_contract_op_si, debug_contract_op_si
from mrh.tests.lassi.addons import fuzz_sivecs, case_matrix_o0_o1
op = (op_o0, op_o1)

def setUpModule ():
    global mol, mf, las, lsi
    xyz = '''He    -0.530330085890  -0.530330085890  -0.530330085890
             He     0.530330085890   0.530330085890  -0.530330085890
             He     0.530330085890  -0.530330085890   0.530330085890
             He    -0.530330085890   0.530330085890   0.530330085890'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, output='/dev/null', verbose=0)
                 #output='test_tetrahedron.log',
                 #verbose=5)
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, [2,2,2,2], [(1,1),(1,1),(1,1),(1,1)], spin_sub=(1,1,1,1))
    mo_coeff = las.localize_init_guess ([[i,] for i in range (4)])
    las.lasci_(mo_coeff)
    lsi = LASSIS (las).run ()

def tearDownModule():
    global mol, mf, las, lsi
    mol.stdout.close ()
    del mol, mf, las, lsi

class KnownValues(unittest.TestCase):

    #@unittest.skip('debugging')
    def test_ham_s2_ovlp (self):
        h1, h2 = ham_2q (lsi, las.mo_coeff, veff_c=None, h2eff_sub=None)[1:]
        nelec_frs = lsi.get_nelec_frs ()
        lbls = ('ham','s2','ovlp')
        mats_o0 = op_o0.ham (las, h1, h2, lsi.ci, nelec_frs)[:3]#, orbsym=orbsym, wfnsym=wfnsym)
        fps_o0 = [lib.fp (mat) for mat in mats_o0]
        mats_o1 = op_o1.ham (las, h1, h2, lsi.ci, nelec_frs)[:3]#, orbsym=orbsym, wfnsym=wfnsym)
        for lbl, mat, fp in zip (lbls, mats_o1, fps_o0):
            with self.subTest(opt=1, matrix=lbl):
                self.assertAlmostEqual (lib.fp (mat), fp, 9)

    #@unittest.skip('debugging')
    def test_rdm12s_slow (self):
        nroots = 2
        si = lsi.si[:,:nroots]
        si_ket = si.copy ()
        si_bra = np.roll (si, 1, axis=1).copy ()
        nelec_frs = lsi.get_nelec_frs ()
        d12_o0 = op_o0.roots_trans_rdm12s (lsi, lsi.ci, nelec_frs, si_bra, si_ket)#, orbsym=orbsym, wfnsym=wfnsym)
        d12_o1 = op_o1.roots_trans_rdm12s (lsi, lsi.ci, nelec_frs, si_bra, si_ket)#, orbsym=orbsym, wfnsym=wfnsym)
        for r in range (2):
            for i in range (nroots):
                with self.subTest (rank=r+1, root=i, opt=1):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i]),
                        lib.fp (d12_o1[r][i]), 9)
                with self.subTest ('single matrix constructor', opt=0, rank=r+1, root=i):
                    d12_o0_test = root_trans_rdm12s (lsi, lsi.ci, si_bra, si_ket, state=i,
                                                     soc=False, break_symmetry=False, opt=0)[r]
                    self.assertAlmostEqual (lib.fp (d12_o0_test), lib.fp (d12_o0[r][i]), 9)
                with self.subTest ('single matrix constructor', opt=1, rank=r+1, root=i):
                    d12_o1_test = root_trans_rdm12s (lsi, lsi.ci, si_bra, si_ket, state=i,
                                                     soc=False, break_symmetry=False, opt=1)[r]
                    self.assertAlmostEqual (lib.fp (d12_o1_test), lib.fp (d12_o0[r][i]), 9)

    #@unittest.skip('debugging')
    def test_davidson (self):
        lsi1 = LASSIS (lsi._las).run (davidson_only=True)
        self.assertAlmostEqual (lsi1.e_roots[0], lsi.e_roots[0], 6)

    #@unittest.skip('debugging')
    def test_fdm1 (self):
        nelec_frs = lsi.get_nelec_frs ()
        nroots = nelec_frs.shape[1]
        make_fdm1 = get_fdm1_maker (las, lsi.ci, nelec_frs, lsi.si[:,0:1])
        for iroot in range (nroots):
            for ifrag in range (4):
                with self.subTest (iroot=iroot, ifrag=ifrag):
                    fdm1 = make_fdm1 (iroot, ifrag)
                    sdm1 = make_sdm1 (lsi, iroot, ifrag, si=lsi.si[:,0:1])
                    self.assertAlmostEqual (lib.fp (fdm1), lib.fp (sdm1), 7)

    @unittest.skip('way too slow for some reason')
    def test_contract_hlas_ci (self):
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        nelec_frs = lsi.get_nelec_frs ()
        case_contract_hlas_ci (self, las, h0, h1, h2, lsi.ci, nelec_frs)

    #@unittest.skip('debugging')
    def test_contract_op_si (self):
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        nelec_frs = lsi.get_nelec_frs ()
        smult_fr = lsi.get_smult_fr ()
        case_contract_op_si (self, las, h1, h2, lsi.ci, nelec_frs, smult_fr=smult_fr)


if __name__ == "__main__":
    print("Full Tests for LASSI he he he he tetrahedron")
    unittest.main()

