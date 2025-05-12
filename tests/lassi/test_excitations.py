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
import sys
import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf, mcscf
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, op_o0, op_o1
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, make_stdm12s
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.lassi.lassis.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver

def array_shape_correction (ci0):
    '''For a list of sequences of CI vectors in the same Hilbert space,
    generate a list in which all but the first element of each sequence
    is discarded.'''
    ci1 = []
    for c in ci0:
        c = np.asarray (c)
        #if c.ndim==3: c = c[0]
        ci1.append (c)
    return ci1

def setUpModule ():
    global mol, mf, lsi, op
    xyz='''He 0 0 0
    H 10 0 0
    H 11 0 0
    H 10 2 0
    H 11 2 0'''
    mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=5, output='test_excitations.log')#0, output='/dev/null')
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
    lsi.kernel ()

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
        charges, spins, smults, wfnsyms = get_space_info (lsi._las)
        dneleca = (spins - charges) // 2
        dnelecb = -(charges + spins) // 2
        dsmults = smults - 1
        nelec = [_unpack_nelec (n) for n in las.nelecas_sub]
        neleca = np.array ([n[0] for n in nelec]) 
        nelecb = np.array ([n[1] for n in nelec]) 
        neleca = dneleca + neleca[None,1:]
        nelecb = dnelecb + nelecb[None,1:]
        lroots = lsi.get_lroots ()
        smults_rf = dsmults + 1

        ci0_ref = array_shape_correction (ci_ref)
        def lassi_ref (ci1, iroot):
            ci1 = array_shape_correction (ci1)
            ci2 = [[ci0_ref[ifrag], ci1[ifrag].copy ()] for ifrag in range (las.nfrags)]
            las1 = LASSCF (mf, (1,2,2), (2,2,2), spin_sub=(1,1,1))
            las1.mo_coeff = las.mo_coeff
            las1.state_average_([1,0],
                charges=[[0,0,0],[0,] + list(charges[iroot])],
                spins=[[0,0,0],[0,] + list(spins[iroot])],
                smults=[[1,1,1],[1,] + list(smults[iroot])],
                wfnsyms=[[0,0,0],[0,0,0]]
            )
            las1.ci = ci2
            ci2 = [[ci_ref[ifrag], ci1[ifrag]] for ifrag in range (las.nfrags)]
            las1.ci = ci2
            lsi1 = LASSI (las1)
            e_roots1, si1 = lsi1.kernel ()
            ham_pq = (si1 * e_roots1[None,:]) @ si1.conj ().T
            w = si1[-1].conj () * si1[-1]
            idx = (w) > 1e-7 # See comment below
            return e_roots1[idx], si1[:,idx]

        h0, h1, h2 = LASSI (las).ham_2q ()
        # In general, the Excitation Solver should return the same energy as LASSI with lroots=1
        # in the excitation rootspace. However, the differentiation between overlapping and
        # orthogonal states breaks down in the limit of weak coupling between the reference and
        # excited rootspace. For the doubly-charge-transferred states, the weight of the root that
        # the VRV solver misses is just barely below 1e-8; that of the root which it catches is
        # about 1e-6. The moral of the story is that we should probably not use the excitation
        # solver for double excitations directly.
        for opt in (0,1):
            psexc = ExcitationPSFCISolver (psref, ci_ref, las.ncas_sub, nelec_ref, 
                                           stdout=mol.stdout, verbose=mol.verbose, opt=opt)
            for iroot in range (1, 5): #lsi._las.nroots):
              with self.subTest (opt=opt, rootspace=iroot):
                for i in range (2):
                    psexc.set_excited_fragment_(1+i, (neleca[iroot,i], nelecb[iroot,i]),
                                                smults[iroot,i])
                nroots = np.amin (lroots[:,iroot])
                conv, energy_tot, ci1 = psexc.kernel (h1, h2, ecore=h0, davidson_only=True,
                                                      nroots=nroots)[:3]
                if (opt==0):
                    for i in range (2):
                        c = np.asarray (ci1[i+1]).reshape (nroots,-1)
                        ovlperr = c.conj () @ c.T - np.eye (nroots)
                        self.assertLess (np.amax (np.abs (ovlperr)), 1e-8)
                self.assertTrue (conv)
                e_roots1, si1 = lassi_ref (ci1, iroot)
                idx_match = np.argmin (np.abs (e_roots1-energy_tot))
                self.assertAlmostEqual (energy_tot, e_roots1[idx_match], 6)
                self.assertEqual (idx_match, 0) # local minimum problems
            # In the no-coupling limit, the Excitation solver should give the same result as the normal
            # ImpureProductStateFCISolver
            #psexc._deactivate_vrv = True # spoof the no-coupling limit
            #for iroot in range (1, lsi._las.nroots):
            #    for i in range (2):
            #        psexc.set_excited_fragment_(1+i, (neleca[iroot,i], nelecb[iroot,i]),
            #                                    smults[iroot,i])
            #    conv, energy_tot, ci1 = psexc.kernel (h1, h2, ecore=h0)
            #    with self.subTest ('no-coupling limit', opt=opt, rootspace=iroot):
            #        self.assertTrue (conv)
            #        self.assertAlmostEqual (energy_tot, lsi._las.e_states[iroot], 8)
            #    conv, energy_tot, ci1 = psexc.kernel (h1, h2, ecore=h0,
            #                                          davidson_only=True)
            #    with self.subTest ('no-coupling limit; davidson only', opt=opt, rootspace=iroot):
            #        self.assertTrue (conv)
            #        self.assertAlmostEqual (energy_tot, lsi._las.e_states[iroot], 8)
                

    def test_multiref (self):
        # Similar to test_cs_excitation, but treating the triplet manifold as the reference
        # Only look at the 4 1e hop states
        las = LASSCF (mf, (1,2,2), (2,(1,1),(1,1)), spin_sub=(1,1,1))
        las.state_average_(weights=[1,0,0],
                           charges=[[0,0,0],]*3,
                           spins=[[0,0,0],[0,-2,2],[0,2,-2]],
                           smults=[[1,3,3],]*3,
                           wfnsyms=[[0,0,0],]*3)
        las.mo_coeff = lsi._las.mo_coeff
        ncsf = las.get_ugg ().ncsf_sub
        las.lasci (lroots=ncsf)
        self.assertAlmostEqual (las.e_states[0], lsi._las.e_states[7])
        self.assertAlmostEqual (las.e_states[1], lsi._las.e_states[9])
        self.assertAlmostEqual (las.e_states[1], lsi._las.e_states[10])
        s0m0 = csf_solver (las.mol, smult=0).set (nelec=tuple([1,1]),charge=0,spin=0,norb=1)
        s1mm = csf_solver (las.mol, smult=3).set (nelec=tuple([0,2]),charge=0,spin=-2,norb=2)
        s1m0 = csf_solver (las.mol, smult=3).set (nelec=tuple([1,1]),charge=0,spin=0,norb=2)
        s1mp = csf_solver (las.mol, smult=3).set (nelec=tuple([2,0]),charge=0,spin=2,norb=2)
        fcisolvers = [[s0m0, s1m0, s1m0],
                      [s0m0, s1mm, s1mp],
                      [s0m0, s1mp, s1mm]]
        psref = [ImpureProductStateFCISolver (fcisolvers[i], stdout=mol.stdout,
                                              verbose=mol.verbose, lroots=ncsf)
                 for i in range (3)]
        ci_ref = las.ci
        nelec_ref = [[1,1] for i in range (3)]
        charges, spins, smults, wfnsyms = get_space_info (lsi._las)
        dneleca = (spins - charges) // 2
        dnelecb = -(charges + spins) // 2
        dsmults = smults - 1
        nelec = [_unpack_nelec (n) for n in las.nelecas_sub]
        neleca = np.array ([n[0] for n in nelec]) 
        nelecb = np.array ([n[1] for n in nelec]) 
        neleca = dneleca + neleca[None,1:]
        nelecb = dnelecb + nelecb[None,1:]
        lroots = lsi.get_lroots ()
        lroots[:,:] = 1
        # Convergence failures if lroots>1 because I'm not state-averaging,
        # but if I state-average then the equality between energy_tot and the
        # ref is broken, so I can only have 1 lroot here
        smults_rf = dsmults + 1

        ci0_ref = [array_shape_correction (c) for c in ci_ref]
        def lassi_ref (ci1, iroot):
            ci1 = array_shape_correction (ci1)
            ci1[0] = ci1[0][0] # This is an uncommon use case
            ci2 = [ci0_ref[ifrag]+[ci1[ifrag].copy (),] for ifrag in range (las.nfrags)]
            las1 = LASSCF (mf, (1,2,2), (2,2,2), spin_sub=(1,1,1))
            las1.mo_coeff = las.mo_coeff
            las1.state_average_([1,0,0,0],
                charges=[[0,0,0],]*3+[[0,] + list(charges[iroot])],
                spins=[[0,0,0],[0,-2,2],[0,2,-2],[0,] + list(spins[iroot])],
                smults=[[1,3,3],]*3+[[1,] + list(smults[iroot])],
                wfnsyms=[[0,0,0],]*4
            )
            las1.ci = ci2
            ci2 = [ci_ref[ifrag]+[ci1[ifrag],] for ifrag in range (las.nfrags)]
            las1.ci = ci2
            lsi1 = LASSI (las1)
            e_roots1, si1 = lsi1.kernel ()
            ham_pq = (si1 * e_roots1[None,:]) @ si1.conj ().T
            w = si1[-1].conj () * si1[-1]
            idx = (w) > 1e-7 
            return e_roots1[idx], si1[:,idx]

        h0, h1, h2 = LASSI (las).ham_2q ()
        for opt in (0,1):
            psexc = ExcitationPSFCISolver (psref, ci_ref, las.ncas_sub, nelec_ref,
                                           stdout=mol.stdout, verbose=mol.verbose, opt=opt)
            for iroot in range (1, 5): 
                for i in range (2):
                    weights = np.ones (lroots[i,iroot]) / lroots[i,iroot]
                    psexc.set_excited_fragment_(1+i, (neleca[iroot,i], nelecb[iroot,i]),
                                                smults[iroot,i], weights=weights)
                conv, energy_tot, ci1 = psexc.kernel (h1, h2, ecore=h0)[:3]
                with self.subTest (rootspace=iroot):
                    self.assertTrue (conv)
                    e_roots1, si1 = lassi_ref (ci1, iroot)
                    idx_match = np.argmin (np.abs (e_roots1-energy_tot))
                    self.assertAlmostEqual (energy_tot, e_roots1[idx_match], 6)
                    self.assertEqual (idx_match, 0) # local minimum problems
                conv, energy_tot, ci1 = psexc.kernel (h1, h2, ecore=h0,
                                                      davidson_only=True)[:3]
                with self.subTest ('davidson only', rootspace=iroot):
                    self.assertTrue (conv)
                    e_roots1, si1 = lassi_ref (ci1, iroot)
                    idx_match = np.argmin (np.abs (e_roots1-energy_tot))
                    self.assertAlmostEqual (energy_tot, e_roots1[idx_match], 6)
                    self.assertEqual (idx_match, 0) # local minimum problems

if __name__ == "__main__":
    print("Full Tests for LASSI excitation constructor")
    unittest.main()

