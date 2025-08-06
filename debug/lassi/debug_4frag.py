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
from mrh.tests.lassi.addons import fuzz_sivecs

def setUpModule ():
    global mol, mf, las, nroots, nelec_frs, si
    # State list contains a couple of different 4-frag interactions
    states  = {'charges': [[0,0,0,0],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1]],
               'spins':   [[0,0,0,0],[1,1,-1,-1],[0,1,-1,0], [1,0,0,-1],[-1,-1,1,1],[0,-1,1,0], [-1,0,0,1]],
               'smults':  [[1,1,1,1],[2,2,2,2],  [1,2,2,1],  [2,1,1,2], [2,2,2,2],  [1,2,2,1],  [2,1,1,2]],
               'wfnsyms': [[0,0,0,0],]*7}
    weights = [1.0,] + [0.0,]*6
    nroots = 7
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
    
    mol = gto.M (atom = xyz, basis='STO-3G', symmetry=False,
        verbose=5, output='debug_4frag.log')
        #verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (2,2,2,2),((1,1),(1,1),(1,1),(1,1)))
    las.state_average_(weights=weights, **states)
    a = list (range (18))
    frags = [a[:5], a[5:9], a[9:13], a[13:18]]
    las.mo_coeff = las.localize_init_guess (frags, mf.mo_coeff)
    las.ci = las.get_init_guess_ci (las.mo_coeff, las.get_h2eff (las.mo_coeff))
    lroots = np.minimum (2, las.get_ugg ().ncsf_sub)
    nelec_frs = np.array (
        [[_unpack_nelec (fcibox._get_nelec (solver, nelecas)) for solver in fcibox.fcisolvers]
         for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
    )
    ndet_frs = np.array (
        [[[cistring.num_strings (las.ncas_sub[ifrag], nelec_frs[ifrag,iroot,0]),
           cistring.num_strings (las.ncas_sub[ifrag], nelec_frs[ifrag,iroot,1])]
          for iroot in range (las.nroots)] for ifrag in range (las.nfrags)]
    )
    np.random.seed (1)
    for ifrag, c in enumerate (las.ci):
        for iroot in range (len (c)):
            lr = lroots[ifrag][iroot]
            ndeta, ndetb = ndet_frs[ifrag][iroot]
            ci = np.random.rand (lr, ndeta, ndetb)
            ci /= linalg.norm (ci.reshape (lr,-1), axis=1)[:,None,None]
            if lr > 1:
                ci = ci.reshape (lr,-1)
                w, v = linalg.eigh (ci.conj () @ ci.T)
                idx = w > 0
                w, v = w[idx], v[:,idx]
                v /= np.sqrt (w)[None,:]
                x = np.dot (v.T, ci)
                u = x.conj () @ ci.T
                Q, R = linalg.qr (u)
                ci = (Q.T @ x).reshape (lr, ndeta, ndetb)
            c[iroot] = ci
    rand_mat = np.random.rand (96,96)
    rand_mat += rand_mat.T
    e, si = linalg.eigh (rand_mat)
    si = lib.tag_array (si, rootsym=las_symm_tuple (las)[0])

def tearDownModule():
    global mol, mf, las, nroots, nelec_frs, si
    mol.stdout.close ()
    del mol, mf, las, nroots, nelec_frs, si

class KnownValues(unittest.TestCase):

    @unittest.skip("debugging")
    def test_stdm12s (self):
        d12_o0 = make_stdm12s (las, opt=0)
        d12_o1 = make_stdm12s (las, opt=1)
        for r in range (2):
            for i, j in product (range (si.shape[0]), repeat=2):
                with self.subTest (rank=r+1, bra=i, ket=j):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i,...,j]),
                        lib.fp (d12_o1[r][i,...,j]), 9)

    @unittest.skip("debugging")
    def test_ham_s2_ovlp (self):
        h1, h2 = ham_2q (las, las.mo_coeff, veff_c=None, h2eff_sub=None)[1:]
        lbls = ('ham','s2','ovlp')
        mats_o0 = op_o0.ham (las, h1, h2, las.ci, nelec_frs)[:3]#, orbsym=orbsym, wfnsym=wfnsym)
        fps_o0 = [lib.fp (mat) for mat in mats_o0]
        mats_o1 = op_o1.ham (las, h1, h2, las.ci, nelec_frs)[:3]#, orbsym=orbsym, wfnsym=wfnsym)
        for lbl, mat, fp in zip (lbls, mats_o1, fps_o0):
            with self.subTest(opt=1, matrix=lbl):
                self.assertAlmostEqual (lib.fp (mat), fp, 9)

    @unittest.skip("debugging")
    def test_rdm12s (self):
        si_ket = si
        si_bra = np.roll (si, 1, axis=1)
        d12_o0 = op_o0.roots_trans_rdm12s (las, las.ci, nelec_frs, si_bra, si_ket)#, orbsym=orbsym, wfnsym=wfnsym)
        d12_o1 = op_o1.roots_trans_rdm12s (las, las.ci, nelec_frs, si_bra, si_ket)#, orbsym=orbsym, wfnsym=wfnsym)
        for r in range (2):
            for i in range (nroots):
                with self.subTest (rank=r+1, root=i, opt=1):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i]),
                        lib.fp (d12_o1[r][i]), 9)
                with self.subTest ('single matrix constructor', opt=0, rank=r+1, root=i):
                    d12_o0_test = root_trans_rdm12s (las, las.ci, si_bra, si_ket, state=i,
                                                     soc=False, break_symmetry=False, opt=0)[r]
                    self.assertAlmostEqual (lib.fp (d12_o0_test), lib.fp (d12_o0[r][i]), 9)
                with self.subTest ('single matrix constructor', opt=1, rank=r+1, root=i):
                    d12_o1_test = root_trans_rdm12s (las, las.ci, si_bra, si_ket, state=i,
                                                     soc=False, break_symmetry=False, opt=1)[r]
                    self.assertAlmostEqual (lib.fp (d12_o1_test), lib.fp (d12_o0[r][i]), 9)

    @unittest.skip("debugging")
    def test_lassis (self):
        las0 = las.get_single_state_las (state=0)
        for ifrag in range (len (las0.ci)):
            las0.ci[ifrag][0] = las0.ci[ifrag][0][0]
        lsi = LASSIS (las0)
        lsi.prepare_states_()
        self.assertTrue (lsi.converged)
        case_lassis_fbf_2_model_state (self, lsi)

    @unittest.skip("debugging")
    def test_lassis_1111 (self):
        xyz='''H 0 0 0
        H 3 0 0
        H 6 0 0
        H 9 0 0'''
        mol1 = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
        mf1 = scf.RHF (mol1).run ()

        las1 = LASSCF (mf1, (1,1,1,1), ((0,1),(1,0),(0,1),(1,0)))
        mo_coeff = las1.localize_init_guess ([[0,],[1,],[2,],[3,]])
        las1.lasci_(mo_coeff)
        for dson in (False,True):
            with self.subTest (davidson_only=dson):
                lsi = LASSIS (las1).set (davidson_only=dson)
                if dson:
                    lsi.prepare_states_()
                    h0, h1, h2 = ham_2q (las1, las1.mo_coeff)
                    case_contract_op_si (self, las1, h1, h2, lsi.ci, lsi.get_nelec_frs ())
                lsi.kernel ()
                self.assertTrue (lsi.converged)
                self.assertAlmostEqual (lsi.e_roots[0], -1.867291372401379, 6)
                case_lassis_fbf_2_model_state (self, lsi)
                case_lassis_fbfdm (self, lsi)

    def test_lassis_slow (self):
        las0 = las.get_single_state_las (state=0)
        for ifrag in range (len (las0.ci)):
            las0.ci[ifrag][0] = las0.ci[ifrag][0][0]
        lsi = LASSIS (las0)
        lsi.prepare_states_()
        for dson in (False,True):
            with self.subTest (davidson_only=dson):
                #if dson:
                #    lsi.prepare_states_()
                #    h0, h1, h2 = ham_2q (las0, las0.mo_coeff)
                #    case_contract_op_si (self, las, h1, h2, lsi.ci, lsi.get_nelec_frs ())
                lsi.eig (davidson_only=dson)
                self.assertTrue (lsi.converged)
                self.assertTrue (lsi.converged_si)
                self.assertAlmostEqual (lsi.e_roots[0], -304.5372586630968, 3)
                case_lassis_fbf_2_model_state (self, lsi)
                #case_lassis_fbfdm (self, lsi)
                lsi.si = fuzz_sivecs (lsi.si)

    @unittest.skip("debugging")
    def test_scallowed (self):
        las0 = las.get_single_state_las (state=0)
        for ifrag in range (len (las0.ci)):
            las0.ci[ifrag][0] = las0.ci[ifrag][0][0]
        lsi = LASSIS (las0)
        lsi.prepare_states_()
        nelec_frs = lsi.get_nelec_frs ()
        exc_test = get_scallowed_interactions (nelec_frs)
        hopping_index = lst_hopping_index (nelec_frs)
        nop = np.abs (hopping_index).sum ((0,1))
        exc_ref = np.asarray (np.where (nop<=4)).T
        self.assertTrue (np.all (exc_test==exc_ref))

    @unittest.skip("debugging")
    def test_fdm1 (self):
        make_fdm1 = get_fdm1_maker (las, las.ci, nelec_frs, si)
        for iroot in range (nroots):
            for ifrag in range (4):
                with self.subTest (iroot=iroot, ifrag=ifrag):
                    fdm1 = make_fdm1 (iroot, ifrag)
                    sdm1 = make_sdm1 (las, iroot, ifrag, si=si)
                    self.assertAlmostEqual (lib.fp (fdm1), lib.fp (sdm1), 7)

    @unittest.skip("debugging")
    def test_contract_hlas_ci (self):
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        case_contract_hlas_ci (self, las, h0, h1, h2, las.ci, nelec_frs)        

    @unittest.skip("debugging")
    def test_contract_op_si (self):
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        case_contract_op_si (self, las, h1, h2, las.ci, nelec_frs)        


if __name__ == "__main__":
    print("Full Tests for LASSI o1 4-fragment intermediates")
    unittest.main()

