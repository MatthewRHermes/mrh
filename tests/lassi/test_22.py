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
from pyscf import lib, gto, scf, mcscf, ao2mo
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, LASSIrq, LASSIrqCT
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, make_stdm12s
from mrh.my_pyscf.lassi.spaces import all_single_excitations, SingleLASRootspace
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi import op_o0, op_o1, lassis

def setUpModule ():
    global mol, mf, lsi, las, mc, op
    xyz='''H 0 0 0
    H 1 0 0
    H 3 0 0
    H 4 0 0'''
    mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()

    # Random Hamiltonian
    rng = np.random.default_rng (424)
    mf._eri = rng.random (mf._eri.shape)
    hcore = rng.random ((4,4))
    hcore = hcore + hcore.T
    mf.get_hcore = lambda *args: hcore

    # LASSCF with CASCI-limit model space
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    las.lasci ()
    las1 = las
    for i in range (2): las1 = all_single_excitations (las1)
    charges, spins, smults, wfnsyms = get_space_info (las1)
    lroots = 4 - smults
    idx = (charges!=0) & (lroots==3)
    lroots[idx] = 1
    las1.conv_tol_grad = las.conv_tol_self = 9e99
    las1.lasci (lroots=lroots.T)
    las1.dump_spaces ()
    lsi = LASSI (las1)
    lsi.kernel (opt=0)

    # CASCI limit
    mc = mcscf.CASCI (mf, 4, 4).run ()

    op = (op_o0, op_o1)

def tearDownModule():
    global mol, mf, lsi, las, mc, op
    mol.stdout.close ()
    del mol, mf, lsi, las, mc, op

class KnownValues(unittest.TestCase):

    def test_op_o1 (self):
        e_roots, si = LASSI (lsi._las).kernel (opt=1)
        self.assertAlmostEqual (lib.fp (e_roots), lib.fp (lsi.e_roots), 8)
        ovlp = si.conj ().T @ lsi.si
        u, svals, vh = linalg.svd (ovlp)
        self.assertAlmostEqual (lib.fp (svals), lib.fp (np.ones (len (svals))), 8)

    def test_casci_limit (self):
        # CASCI limit
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)

        # LASSI in the CASCI limit
        las, e_roots, si = lsi._las, lsi.e_roots, lsi.si
        with self.subTest ("total energy"):
            self.assertAlmostEqual (e_roots[0], mc.e_tot, 8)
        for opt in range (2):
            with self.subTest (opt=opt):
                lasdm1s, lasdm2s = root_make_rdm12s (las, las.ci, si, state=0, opt=opt)
                lasdm1 = lasdm1s.sum (0)
                lasdm2 = lasdm2s.sum ((0,3))
                with self.subTest ("casdm1"):
                    self.assertAlmostEqual (lib.fp (lasdm1), lib.fp (casdm1), 8)
                with self.subTest ("casdm2"):
                    self.assertAlmostEqual (lib.fp (lasdm2), lib.fp (casdm2), 8)
                stdm1s = make_stdm12s (las, opt=opt)[0][9:13,:,:,:,9:13] # second rootspace
                with self.subTest("state indexing"):
                    # column-major ordering for state excitation quantum numbers:
                    # earlier fragments advance faster than later fragments
                    self.assertAlmostEqual (lib.fp (stdm1s[0,:,:2,:2,0]),
                                            lib.fp (stdm1s[2,:,:2,:2,2]))
                    self.assertAlmostEqual (lib.fp (stdm1s[0,:,2:,2:,0]),
                                            lib.fp (stdm1s[1,:,2:,2:,1]))

    def test_lassirq (self):
        lsi1 = LASSIrq (las, 2, 3).run ()
        self.assertAlmostEqual (lsi1.e_roots[0], mc.e_tot, 8)

    def test_lassirqct (self):
        lsi1 = LASSIrqCT (las, 2, 3).run ()
        self.assertAlmostEqual (lsi1.e_roots[0], -4.2879945248402445, 8)

    def test_contract_hlas_ci (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q ()
        nelec = lsi.get_nelec_frs ()
        ci_fr = las.ci
        ham = (si * (e_roots[None,:]-h0)) @ si.conj ().T
        ndim = len (e_roots)        

        spaces = [SingleLASRootspace (las, m, s, c, 0) for c,m,s,w in zip (*get_space_info (las))]

        lroots = lsi.get_lroots ()
        lroots_prod = np.prod (lroots, axis=0)
        nj = np.cumsum (lroots_prod)
        ni = nj - lroots_prod
        for opt in range (2):
            hket_fr_pabq = op[opt].contract_ham_ci (las, h1, h2, ci_fr, nelec, ci_fr, nelec)
            for f, (ci_r, hket_r_pabq) in enumerate (zip (ci_fr, hket_fr_pabq)):
                current_order = list (range (las.nfrags-1, -1, -1)) + [las.nfrags]
                current_order.insert (0, current_order.pop (f))
                for r, (ci, hket_pabq) in enumerate (zip (ci_r, hket_r_pabq)):
                    if ci.ndim < 3: ci = ci[None,:,:]
                    proper_shape = np.append (lroots[:,r], ndim)
                    current_shape = proper_shape[current_order]
                    to_proper_order = list (np.argsort (current_order))
                    hket_pq = lib.einsum ('rab,pabq->rpq', ci.conj (), hket_pabq)
                    hket_pq = hket_pq.reshape (current_shape)
                    hket_pq = hket_pq.transpose (*to_proper_order)
                    hket_pq = hket_pq.reshape ((lroots_prod[r], ndim))
                    hket_ref = ham[ni[r]:nj[r]]
                    for s, (k, l) in enumerate (zip (ni, nj)):
                        hket_pq_s = hket_pq[:,k:l]
                        hket_ref_s = hket_ref[:,k:l]
                        # TODO: opt=1 for things other than single excitation
                        if opt==1 and not spaces[r].is_single_excitation_of (spaces[s]): continue
                        #elif opt==1: print (r,s, round (lib.fp (hket_pq_s)-lib.fp (hket_ref_s),3))
                        with self.subTest (opt=opt, frag=f, bra_space=r, ket_space=s):
                            self.assertAlmostEqual (lib.fp (hket_pq_s), lib.fp (hket_ref_s), 8)

    def test_lassis (self):
        for opt in (0,1):
            with self.subTest (opt=opt):
                lsis = lassis.LASSIS (las).run (opt=opt)
                e_upper = las.e_states[0]
                e_lower = lsi.e_roots[0]
                self.assertLessEqual (e_lower, lsis.e_roots[0])
                self.assertLessEqual (lsis.e_roots[0], e_upper)
                self.assertEqual (len (lsis.e_roots), 20)
                # Reference depends on rng seed obviously b/c this is not casci limit
                self.assertAlmostEqual (lsis.e_roots[0], -4.134472877702426, 8)

if __name__ == "__main__":
    print("Full Tests for LASSI of random 2,2 system")
    unittest.main()

