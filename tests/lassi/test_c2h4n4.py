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
import os
import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import roots_make_rdm12s, root_make_rdm12s, make_stdm12s, ham_2q
from mrh.my_pyscf.lassi import LASSI, LASSIS
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_lassis_fbf_2_model_state
from mrh.tests.lassi.addons import case_lassis_fbfdm, case_contract_op_si
topdir = os.path.abspath (os.path.join (__file__, '..'))

def setUpModule ():
    global las, lsi, rdm1s, rdm2s
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = 0 #lib.logger.DEBUG
    mol.output = '/dev/null' #'test_c2h4n4.log'
    mol.spin = 0
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
    las.state_average_(weights=[1.0/5.0,]*5,
        spins=[[0,0],[0,0],[2,-2],[-2,2],[2,2]],
        smults=[[1,1],[3,3],[3,3],[3,3],[3,3]])
    las.frozen = list (range (las.mo_coeff.shape[-1]))
    ugg = las.get_ugg ()
    las.mo_coeff = np.loadtxt (os.path.join (topdir, 'test_c2h4n4_mo.dat'))
    las.ci = ugg.unpack (np.loadtxt (os.path.join (topdir, 'test_c2h4n4_ci.dat')))[1]
    lroots = 2 * np.ones ((2,7), dtype=int)
    lroots[:,1] = 1 # <- that rootspace is responsible for spin contamination
    las.conv_tol_grad = 1e-8
    las.lasci (lroots=lroots)
    # TODO: potentially save and load CI vectors
    # requires extending features of ugg
    #las.set (conv_tol_grad=1e-8).run ()
    #np.savetxt ('test_c2h4n4_mo.dat', las.mo_coeff)
    #np.savetxt ('test_c2h4n4_ci.dat', ugg.pack (las.mo_coeff, las.ci))
    #las.e_states = las.energy_nuc () + las.states_energy_elec ()
    lsi = LASSI (las).run ()
    rdm1s, rdm2s = roots_make_rdm12s (las, las.ci, lsi.si)
    las = LASSCF (mf, (4,2,4), ((4,0),(1,1),(0,4)), spin_sub=(5,3,5))
    ci_quin = np.array ([[1.0,],])
    ci_sing = np.diag ([1,-1])[::-1] / np.sqrt (2)
    las.ci = [[ci_quin], [ci_sing], [ci_quin.copy ()]]
    las.e_states = las.energy_nuc () + las.states_energy_elec ()

def tearDownModule():
    global las, lsi, rdm1s, rdm2s
    mol = lsi._las.mol
    mol.stdout.close ()
    del las, lsi, rdm1s, rdm2s

class KnownValues(unittest.TestCase):
    def test_evals (self):
        self.assertAlmostEqual (lib.fp (lsi.e_roots), 71.48678303162376, 6)

    def test_si (self):
        # Arbitrary signage in both the SI and CI vector, sadly
        # Actually this test seems really inconsistent overall...
        dens = lsi.si.conj () * lsi.si
        self.assertAlmostEqual (lib.fp (np.diag (dens)), 1.2321935065030327, 4)

    def test_nelec (self):
        for ix, ne in enumerate (lsi.nelec):
          with self.subTest(ix):
            if ix in (1,5,8,11):
                self.assertEqual (ne, (6,2))
            else:
                self.assertEqual (ne, (4,4))

    def test_s2 (self):
        s2_array = np.zeros (16)
        quintets = [1,2,5,8,11]
        for ix in quintets: s2_array[ix] = 6
        triplets = [3,6,7,9,10,12,13]
        for ix in triplets: s2_array[ix] = 2
        self.assertAlmostEqual (lib.fp (lsi.s2), lib.fp (s2_array), 3)

    def test_tdms (self):
        las, si = lsi._las, lsi.si
        stdm1s, stdm2s = make_stdm12s (las)
        nelec = float (sum (las.nelecas))
        for ix in range (stdm1s.shape[0]):
            d1 = stdm1s[ix,...,ix].sum (0)
            d2 = stdm2s[ix,...,ix].sum ((0,3))
            with self.subTest (root=ix):
                self.assertAlmostEqual (np.trace (d1), nelec,  9)
                self.assertAlmostEqual (np.einsum ('ppqq->',d2), nelec*(nelec-1), 9)
        rdm1s_test = np.einsum ('ar,asijb,br->rsij', si.conj (), stdm1s, si) 
        rdm2s_test = np.einsum ('ar,asijtklb,br->rsijtkl', si.conj (), stdm2s, si) 
        self.assertAlmostEqual (lib.fp (rdm1s_test), lib.fp (rdm1s), 9)
        self.assertAlmostEqual (lib.fp (rdm2s_test), lib.fp (rdm2s), 9)

    def test_rdms (self):
        las, e_roots = lsi._las, lsi.e_roots
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        d1_r = rdm1s.sum (1)
        d2_r = rdm2s.sum ((1, 4))
        nelec = float (sum (las.nelecas))
        for ix, (d1, d2) in enumerate (zip (d1_r, d2_r)):
            with self.subTest (root=ix):
                self.assertAlmostEqual (np.trace (d1), nelec,  9)
                self.assertAlmostEqual (np.einsum ('ppqq->',d2), nelec*(nelec-1), 9)
        e_roots_test = h0 + np.tensordot (d1_r, h1, axes=2) + np.tensordot (d2_r, h2, axes=4) / 2
        for e1, e0 in zip (e_roots_test, e_roots):
            self.assertAlmostEqual (e1, e0, 8)

    def test_singles_constructor (self):
        from mrh.my_pyscf.lassi.spaces import all_single_excitations
        las2 = all_single_excitations (lsi._las)
        las2.check_sanity ()
        # Meaning of tuple: (na+nb,smult)
        # from state 0, (1+2,2)&(3+2,2) & a<->b & l<->r : 4 states
        # from state 1, smults=((2,4),(4,2),(4,4)) permutations of above: 12 additional states
        # from states 2 and 3, (4+1,4)&(0+3,4) & a<->b & l<->r : 4 additional states
        # from state 4, (2+1,2)&(4+1,4) & (2+1,4)&(4+1,4) 
        #             & (3+0,4)&(3+2,2) & (3+0,4)&(3+2,4) & l<->r : 8 additional states
        # 5 + 4 + 12 + 4 + 8 = 33
        self.assertEqual (las2.nroots, 33)

    def test_spin_shuffle (self):
        from mrh.my_pyscf.lassi.spaces import spin_shuffle, spin_shuffle_ci
        mf = lsi._las._scf
        las3 = spin_shuffle (las)
        las3.check_sanity ()
        # The number of states is the number of graphs connecting one number
        # in each row which sum to zero:
        # -2 -1 0 +1 +2
        #    -1 0 +1
        # -2 -1 0 +1 +2
        # For the first two rows, paths which sum to -3 and +3 are immediately
        # excluded. Two paths connecting the first two rows each sum to -2 and +2
        # and three paths each sum to -1, 0, +1. Each partial sum then has one
        # remaining option to complete the path, so
        # 2 + 3 + 3 + 3 + 2 = 13
        with self.subTest ("state construction"):
            self.assertEqual (las3.nroots, 13)
        las3.ci = spin_shuffle_ci (las3, las3.ci)
        lsi2 = LASSI (las3).run ()
        errvec = lsi2.s2 - np.around (lsi2.s2)
        with self.subTest ("CI vector rotation"):
            self.assertLess (np.amax (np.abs (errvec)), 1e-8)

    def test_lassis (self):
        for opt in (0,1):
            with self.subTest (opt=opt):
                las1 = LASSCF (las._scf, (4,4), (4,4), spin_sub=(1,1))
                las1.mo_coeff = las.mo_coeff
                las1.lasci ()
                lsis = LASSIS (las1).run (opt=opt, max_cycle_macro=1)
                case_lassis_fbf_2_model_state (self, lsis)
                case_lassis_fbfdm (self, lsis)
        with self.subTest ('davidson_only'):
            las1 = LASSCF (las._scf, (4,4), (4,4), spin_sub=(1,1))
            las1.mo_coeff = las.mo_coeff
            las1.lasci ()
            lsis = LASSIS (las1).run (davidson_only=True, max_cycle_macro=1)


    def test_lassis_slow (self):
        # TODO: optimize implementation and eventually merge with test_lassis
        mol = struct (2.0, 2.0, '6-31g', symmetry=False)
        mol.output = '/dev/null'
        mol.verbose = 0
        mol.spin = 8
        mol.build ()
        mf = scf.RHF (mol).run () 
        las1 = LASSCF (mf, (5,5), ((3,2),(2,3)), spin_sub=(2,2))
        mo_coeff = las1.localize_init_guess ((list (range (5)), list (range (5,10))))
        las1.kernel (mo_coeff)
        for dson in (False, True):
            for opt in (0,1):
                with self.subTest (opt=opt, davidson_only=dson):
                    lsis = LASSIS (las1).run (opt=opt, davidson_only=dson)
                    self.assertTrue (lsis.converged)
                    self.assertAlmostEqual (lsis.e_roots[0], -295.52185731568903, 7)
                    case_lassis_fbf_2_model_state (self, lsis)
                    case_lassis_fbfdm (self, lsis)
            with self.subTest ('as_scanner', davidson_only=dson):
                lsis_scanner = lsis.as_scanner ()
                mol2 = struct (1.9, 1.9, '6-31g', symmetry=False)
                mol2.verbose = 0
                mol2.output = '/dev/null'
                mol2.build ()
                lsis_scanner (mol2)
                self.assertTrue (lsis_scanner.converged)
                mf2 = scf.RHF (mol2).run ()
                las2 = LASSCF (mf2, (5,5), ((3,2),(2,3)), spin_sub=(2,2))
                las2.mo_coeff = lsis_scanner.mo_coeff
                las2.lasci ()
                lsis2 = LASSIS (las2).run (davidson_only=dson)
                self.assertTrue (lsis2.converged)
                self.assertAlmostEqual (lsis_scanner.e_roots[0], lsis2.e_roots[0], 5)


    def test_contract_hlas_ci (self):
        las, nelec_frs = lsi._las, lsi.get_nelec_frs ()
        h0, h1, h2 = lsi.ham_2q ()
        ci = [c[:4] for c in las.ci] # No SOC yet
        nelec_frs = nelec_frs[:,:4,:] # No SOC yet
        case_contract_hlas_ci (self, las, h0, h1, h2, ci, nelec_frs)

    def test_contract_op_si (self):
        las, nelec_frs = lsi._las, lsi.get_nelec_frs ()
        h0, h1, h2 = lsi.ham_2q ()
        ci = [c[:4] for c in las.ci] # No SOC yet
        nelec_frs = nelec_frs[:,:4,:] # No SOC yet
        case_contract_op_si (self, las, h1, h2, ci, nelec_frs)

if __name__ == "__main__":
    print("Full Tests for SA-LASSI of c2h4n4 molecule")
    unittest.main()

