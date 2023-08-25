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
from pyscf import lib, gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, op_o0, op_o1
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, make_stdm12s
from mrh.my_pyscf.lassi.states import all_single_excitations
from mrh.my_pyscf.lassi.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver

def setUpModule ():
    global mol, mf, lsi, op
    xyz='''He 0 0 0
    H 10 0 0
    H 11 0 0
    H 10 2 0
    H 11 2 0'''
    mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()

    # LASSCF with CASCI-limit model space
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    mo_coeff = las.localize_init_guess ([[1,2],[3,4]])
    las.kernel (mo_coeff)
    for i in range (2): las = all_single_excitations (las)
    charges, spins, smults, wfnsyms = get_space_info (las)
    lroots = 4 - smults
    idx = (charges!=0) & (lroots==3)
    lroots[idx] = 1
    las.lasci (lroots=lroots.T)
    lsi = LASSI (las)
    lsi.kernel (opt=0)

    op = (op_o0, op_o1)

def tearDownModule():
    global mol, mf, lsi, op
    mol.stdout.close ()
    del mol, mf, lsi, op

class KnownValues(unittest.TestCase):

    def test_cs_excitation (self):
        las = LASSCF (mf, (1,2,2), (2,2,2), spin_sub=(1,1,1))
        las.mo_coeff = lsi._las.mo_coeff
        ncsf = las.get_ugg ().ncsf_sub
        las.lasci (lroots=ncsf)
        self.assertAlmostEqual (las.e_states[0], lsi._las.e_states[0])
        psref = ImpureProductStateFCISolver ([b.fcisolvers[0] for b in las.fciboxes],
                                             stdout=mol.stdout, verbose=mol.verbose, lroots=ncsf)
        ci_ref = [c[0] for c in las.ci]
        nelec_ref = [[1,1] for i in range (3)]
        psexc = ExcitationPSFCISolver (psref, ci_ref, las.ncas_sub, nelec_ref)
        charges, spins, smults, wfnsyms = get_space_info (lsi._las)
        dneleca = (spins - charges) // 2
        dnelecb = -(charges + spins) // 2
        dsmults = smults - 1
        lroots = lsi.get_lroots ()
        smults_rf = dsmults + 1

        # TODO: remove the ci0 kwarg from the excitation solver and implement
        # get_init_guess in productstate solver. 
        h0, h1, h2 = LASSI (las).ham_2q ()
        for iroot in range (1, lsi._las.nroots):
          with self.subTest (rootspace=iroot):
            for i in range (2):
                weights = np.zeros (lroots[i,iroot])
                weights[0] = 1
                psexc.set_excited_fragment_(1+i, dneleca[iroot,i], dnelecb[iroot,i],
                                            dsmults[iroot,i], weights=weights)
            conv, energy_tot, ci1 = psexc.kernel (h1, h2, ecore=h0)
            self.assertTrue (conv)
            self.assertAlmostEqual (energy_tot, lsi._las.e_states[iroot], 8)

if __name__ == "__main__":
    print("Full Tests for LASSI excitation constructor")
    unittest.main()

