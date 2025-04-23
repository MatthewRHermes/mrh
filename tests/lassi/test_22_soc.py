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
from pyscf import lib, gto, scf, mcscf, fci, ao2mo
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, LASSIrq, LASSIrqCT
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, roots_trans_rdm12s, make_stdm12s
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi import op_o0, op_o1, lassis
from mrh.my_pyscf.lassi.op_o1 import get_fdm1_maker
from mrh.my_pyscf.lassi.sitools import make_sdm1
from mrh.tests.lassi.addons import case_contract_hlas_ci, case_contract_op_si

def setUpModule ():
    global mol, mf, lsi, las, mc, op, old_compute_hso
    xyz='''H 0 0 0
    H 1 0 0
    H 3 0 0
    H 4 0 0'''
    mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()

    # Random Hamiltonian
    rng = np.random.default_rng ()
    mf._eri = rng.random (mf._eri.shape)
    hcore = rng.random ((4,4))
    hcore = hcore + hcore.T
    mf.get_hcore = lambda *args: hcore
    hso = rng.random ((3,4,4))
    hso = hso + hso.transpose (0,2,1)
    from mrh.my_pyscf.mcscf import soc_int
    old_compute_hso = soc_int.compute_hso
    soc_int.compute_hso = lambda *args, **kwargs: hso

    # LASSCF with CASCI-limit model space
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    las.lasci ()
    # TODO: automate this state selection
    las1 = las.state_average (
        weights = [1,] + [0 for i in range (15)],
        spins  = [[0,0], [-2,0], [0,0], [2,0],
                  [0,-2],[-2,-2],[0,-2],[2,-2],
                  [0,0], [-2,0], [0,0], [2,0],
                  [0,2], [-2,2], [0,2], [2,2]],
        smults = [[1,1],[3,1],[3,1],[3,1],
                  [1,3],[3,3],[3,3],[3,3],
                  [1,3],[3,3],[3,3],[3,3],
                  [1,3],[3,3],[3,3],[3,3]])
    for i in range (2): las1 = all_single_excitations (las1)
    las1.conv_tol_grad = las.conv_tol_self = 9e99
    las1.lasci (lroots=las1.get_ugg ().ncsf_sub)
    #las1.dump_spaces ()
    lsi = LASSI (las1, soc=1, break_symmetry=True, opt=0)
    lsi.kernel ()

    # CASCI limit (I have to spoof this sadly)
    mc = mcscf.CASCI (mf, 4, 4).run ()
    mc.ncas = 8
    mc.nelecas = (4,0)
    h0, h1, h2_ = lsi.ham_2q (soc=1)
    h2 = np.zeros ((8,8,8,8), dtype=h1.dtype)
    h2[:4,:4,:4,:4] = h2[4:,4:,:4,:4] = h2[:4,:4,4:,4:] = h2[4:,4:,4:,4:] = h2_[:]
    mc.fcisolver = fci.fci_dhf_slow.FCI()
    mc.e_tot, mc.ci = mc.fcisolver.kernel (h1, h2, 8, nelec=4, ecore=h0)
    dm1s, dm2s_ = mc.fcisolver.make_rdm12 (mc.ci, 8, 4)
    dm2s = np.zeros ((2,4,4,2,4,4), dtype=dm2s_.dtype)
    dm2s[0,:,:,0,:,:] = dm2s_[:4,:4,:4,:4]
    dm2s[1,:,:,0,:,:] = dm2s_[4:,4:,:4,:4]
    dm2s[0,:,:,1,:,:] = dm2s_[:4,:4,4:,4:]
    dm2s[1,:,:,1,:,:] = dm2s_[4:,4:,4:,4:]
    mc.fcisolver.dm1s = dm1s
    mc.fcisolver.dm2s = dm2s
    e_check = h0 + np.tensordot (dm1s, h1, axes=2) + .5*np.tensordot (dm2s_, h2, axes=4)
    assert (abs (e_check.imag) < 1e-8)
    assert (abs (e_check.real - mc.e_tot) < 1e-8)

    op = (op_o0, op_o1)

def tearDownModule():
    global mol, mf, lsi, las, mc, op, old_compute_hso
    from mrh.my_pyscf.mcscf import soc_int
    soc_int.compute_hso = old_compute_hso
    mol.stdout.close ()
    del mol, mf, lsi, las, mc, op, old_compute_hso

class KnownValues(unittest.TestCase):

    def test_op_o1 (self):
        lsi1 = LASSI (lsi._las, opt=1, soc=1, break_symmetry=True).run ()
        with self.subTest ("total energy"):
            self.assertAlmostEqual (lib.fp (lsi1.e_roots), lib.fp (lsi.e_roots), 8)
        ovlp = lsi1.si.conj ().T @ lsi.si
        u, svals, vh = linalg.svd (ovlp)
        with self.subTest ("sivec space"):
            self.assertAlmostEqual (lib.fp (svals), lib.fp (np.ones (len (svals))), 8)
        ham_o0 = (lsi.si * lsi.e_roots[None,:]) @ lsi.si.conj ().T
        ham_o1 = (lsi1.si * lsi1.e_roots[None,:]) @ lsi1.si.conj ().T
        with self.subTest ("Hamiltonian including phase"):
            self.assertAlmostEqual (lib.fp (ham_o0), lib.fp (ham_o1), 8)

    def test_casci_limit (self):
        # CASCI limit
        casdm1s, casdm2s = mc.fcisolver.dm1s, mc.fcisolver.dm2s

        # LASSI in the CASCI limit
        las, e_roots, si = lsi._las, lsi.e_roots, lsi.si
        with self.subTest ("total energy"):
            self.assertAlmostEqual (e_roots[0], mc.e_tot, 8)
        for opt in range (2):
            with self.subTest (opt=opt):
                lasdm1s, lasdm2s = root_make_rdm12s (las, las.ci, si, state=0, opt=opt)
                with self.subTest ("casdm1s"):
                    self.assertAlmostEqual (lib.fp (lasdm1s), lib.fp (casdm1s), 4)
                with self.subTest ("casdm2s"):
                    self.assertAlmostEqual (lib.fp (lasdm2s), lib.fp (casdm2s), 4)

    def test_si_trans_rdm12s (self):
        las, e_roots, si_ket = lsi._las, lsi.e_roots, lsi.si
        si_bra = np.roll (si_ket, 1, axis=1)
        stdm1s, stdm2s = make_stdm12s (las, soc=True, opt=1)
        rdm1s_ref = lib.einsum ('ir,jr,iabj->rab', si_bra.conj (), si_ket, stdm1s)
        rdm2s_ref = lib.einsum ('ir,jr,isabtcdj->rsabtcd', si_bra, si_ket.conj (), stdm2s)
        for opt in range (2):
            with self.subTest (opt=opt):
                lasdm1s, lasdm2s = roots_trans_rdm12s (las, las.ci, si_bra, si_ket, opt=opt)
                with self.subTest ("lasdm1s"):
                    self.assertAlmostEqual (lib.fp (lasdm1s), lib.fp (rdm1s_ref), 8)
                with self.subTest ("lasdm2s"):
                    self.assertAlmostEqual (lib.fp (lasdm2s), lib.fp (rdm2s_ref), 8)

    def test_davidson (self):
        lsi1 = LASSI (lsi._las, soc=1, break_symmetry=True).run (davidson_only=True)
        self.assertAlmostEqual (lsi1.e_roots[0], lsi.e_roots[0], 8)
        ovlp = np.dot (lsi1.si[:,0], lsi.si[:,0].conj ())
        ovlp = ovlp.conj () * ovlp
        self.assertAlmostEqual (ovlp, 1.0, 4)

    def test_contract_op_si (self):
        e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
        h0, h1, h2 = lsi.ham_2q (soc=1)
        case_contract_op_si (self, las, h1, h2, las.ci, lsi.get_nelec_frs (), soc=1)

    #def test_lassirq (self):
    #    lsi1 = LASSIrq (las, 2, 3).run ()
    #    self.assertAlmostEqual (lsi1.e_roots[0], mc.e_tot, 8)

    #def test_lassirqct (self):
    #    lsi1 = LASSIrqCT (las, 2, 3).run ()
    #    self.assertAlmostEqual (lsi1.e_roots[0], -4.2879945248402445, 8)

    #def test_contract_hlas_ci (self):
    #    e_roots, si, las = lsi.e_roots, lsi.si, lsi._las
    #    h0, h1, h2 = lsi.ham_2q ()
    #    case_contract_hlas_ci (self, las, h0, h1, h2, las.ci, lsi.get_nelec_frs ())

    #def test_lassis (self):
    #    for opt in (0,1):
    #        with self.subTest (opt=opt):
    #            lsis = lassis.LASSIS (las).run (opt=opt)
    #            e_upper = las.e_states[0]
    #            e_lower = lsi.e_roots[0]
    #            self.assertLessEqual (e_lower, lsis.e_roots[0])
    #            self.assertLessEqual (lsis.e_roots[0], e_upper)
    #            self.assertEqual (len (lsis.e_roots), 20)
    #            # Reference depends on rng seed obviously b/c this is not casci limit
    #            self.assertAlmostEqual (lsis.e_roots[0], -4.134472877702426, 8)

    #def test_fdm1 (self):
    #    make_fdm1 = get_fdm1_maker (lsi, lsi.ci, lsi.get_nelec_frs (), lsi.si)
    #    for iroot in range (lsi.nroots):
    #        for ifrag in range (lsi.nfrags):
    #            with self.subTest (iroot=iroot, ifrag=ifrag):
    #                fdm1 = make_fdm1 (iroot, ifrag)
    #                sdm1 = make_sdm1 (lsi, iroot, ifrag)
    #                self.assertAlmostEqual (lib.fp (fdm1), lib.fp (sdm1), 7)

if __name__ == "__main__":
    print("Full Tests for LASSI of random 2,2 system")
    unittest.main()

