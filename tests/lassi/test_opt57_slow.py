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
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi.lassi import roots_make_rdm12s, make_stdm12s, ham_2q
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_contract_op_si

op = (op_o0, op_o1)

def setUpModule ():
    global mol, mf, las, nstates, nelec_frs, si, orbsym, wfnsym
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
    lroots = np.ones ((3, 57), dtype=int)
    offs = [0,]
    for field in ('charges', 'spins', 'smults', 'wfnsyms'):
        states6[field] = [[row[1], row[2], row[0]] for row in states5[field]]
        states7[field] = [[row[2], row[0], row[1]] for row in states5[field]]
    for d in [states1, states2, states3, states4, states5, states6, states7]:
        offs.append (len (states['charges']))
        for field in ('charges', 'spins', 'smults', 'wfnsyms'):
            states[field] = states[field] + d[field]
    lroots[:,offs[0]] = [2, 2, 2]
    lroots[:,offs[1]] = [2, 2, 2]
    lroots[:,offs[2]] = [2, 1, 2]
    lroots[:,offs[3]] = [2, 1, 2]
    lroots[:,offs[4]] = [2, 2, 2]
    lroots[:,offs[5]] = [2, 2, 2]
    weights = [1.0,] + [0.0,]*56
    nroots = 57
    nstates = 91
    # End building crazy state list
    
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry='Cs')
    mol.verbose = 0 #lib.logger.INFO 
    mol.output = '/dev/null' #'test_lassi_op.log'
    mol.spin = 0 
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,2,4), (4,2,4))
    las.state_average_(weights=weights, **states)
    las.mo_coeff = las.localize_init_guess ((list (range (3)),
        list (range (3,7)), list (range (7,10))), mf.mo_coeff)
    las.ci = las.get_init_guess_ci (las.mo_coeff, las.get_h2eff (las.mo_coeff))
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
    for iroot in range (las.nroots):
        for ifrag in range (las.nfrags):
            lroots_r = lroots[ifrag,iroot]
            ndet_s = ndet_frs[ifrag,iroot]
            ci = np.random.rand (lroots_r, ndet_s[0], ndet_s[1])
            ci /= linalg.norm (ci.reshape (lroots_r,-1), axis=1)[:,None,None]
            if lroots_r==1:
                ci=ci[0]
            else:
                ci = ci.reshape (lroots_r,-1)
                w, v = linalg.eigh (ci.conj () @ ci.T)
                idx = w > 0
                w, v = w[idx], v[:,idx]
                v /= np.sqrt (w)[None,:]
                ci = np.dot (v.T, ci).reshape (lroots_r, ndet_s[0], ndet_s[1])
            las.ci[ifrag][iroot] = ci
    orbsym = getattr (las.mo_coeff, 'orbsym', None)
    if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
        orbsym = las.label_symmetry_(las.mo_coeff).orbsym
    if orbsym is not None:
        orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    wfnsym = 0
    #las.lasci (lroots=lroots)
    rand_mat = np.random.rand (nstates,nstates)
    rand_mat += rand_mat.T
    e, si = linalg.eigh (rand_mat)

def tearDownModule():
    global mol, mf, las, nstates, nelec_frs, si, orbsym, wfnsym
    mol.stdout.close ()
    del mol, mf, las, nstates, nelec_frs, si, orbsym, wfnsym

class KnownValues(unittest.TestCase):
    def test_stdm12s (self):
        t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        d12_o0 = make_stdm12s (las, opt=0)
        t1, w1 = lib.logger.process_clock (), lib.logger.perf_counter ()
        d12_o1 = make_stdm12s (las, opt=1)
        t2, w2 = lib.logger.process_clock (), lib.logger.perf_counter ()
        #print (t1-t0, t2-t1)
        #print (w1-w0, w2-w1)
        rootaddr, fragaddr = get_rootaddr_fragaddr (get_lroots (las.ci))
        for r in range (2):
            for i, j in product (range (nstates), repeat=2):
                with self.subTest (rank=r+1, idx=(i,j), spaces=(rootaddr[i], rootaddr[j]),
                                   envs=(list(fragaddr[:,i]),list(fragaddr[:,j]))):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i,...,j]),
                        lib.fp (d12_o1[r][i,...,j]), 9)

    def test_ham_s2_ovlp (self):
        h1, h2 = ham_2q (las, las.mo_coeff, veff_c=None, h2eff_sub=None)[1:]
        lbls = ('ham','s2','ovlp')
        t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        mats_o0 = op_o0.ham (las, h1, h2, las.ci, nelec_frs, orbsym=orbsym, wfnsym=wfnsym)[:3]
        t1, w1 = lib.logger.process_clock (), lib.logger.perf_counter ()
        mats_o1 = op_o1.ham (las, h1, h2, las.ci, nelec_frs, orbsym=orbsym, wfnsym=wfnsym)[:3]
        t2, w2 = lib.logger.process_clock (), lib.logger.perf_counter ()
        #print (t1-t0, t2-t1)
        #print (w1-w0, w2-w1)
        fps_o0 = [lib.fp (mat) for mat in mats_o0]
        for lbl, mat, fp in zip (lbls, mats_o1, fps_o0):
            with self.subTest(matrix=lbl):
                self.assertAlmostEqual (lib.fp (mat), fp, 9)

    def test_rdm12s (self):
        si_bra = si
        si_ket = np.roll (si, 1, axis=1)
        t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        d12_o0 = op_o0.roots_trans_rdm12s (las, las.ci, nelec_frs, si_bra, si_ket, orbsym=orbsym,
                                           wfnsym=wfnsym)
        t1, w1 = lib.logger.process_clock (), lib.logger.perf_counter ()
        d12_o1 = op_o1.roots_trans_rdm12s (las, las.ci, nelec_frs, si_bra, si_ket, orbsym=orbsym,
                                           wfnsym=wfnsym)
        t2, w2 = lib.logger.process_clock (), lib.logger.perf_counter ()
        #print (t1-t0, t2-t1, t3-t2)
        #print (w1-w0, w2-w1, w3-w2)
        for r in range (2):
            for i in range (nstates):
                with self.subTest (rank=r+1, root=i, opt=1):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i]),
                        lib.fp (d12_o1[r][i]), 9)

    def test_contract_hlas_ci (self):
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        case_contract_hlas_ci (self, las, h0, h1, h2, las.ci, nelec_frs)

    def test_contract_op_si (self):
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        case_contract_op_si (self, las, h1, h2, las.ci, nelec_frs)



if __name__ == "__main__":
    print("Full Tests for LASSI matrix elements of 57-space (91-state) manifold")
    unittest.main()

