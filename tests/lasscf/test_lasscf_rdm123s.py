#!/usr/bin/env python
#
# Author: Shreya Verma <shreyav@uchicago.edu>

# The tests are performed for the following purposes:
#   1. Check accuracy of spin-summed LASSCF 2-RDMs against spin-summed 2-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   2. Check accuracy of spin-separated LASSCF 2-RDMs against spin-separated 2-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   3. Check accuracy of spin-summed LASSCF 3-RDMs against spin-summed 3-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   4. Check accuracy of spin-separated LASSCF 3-RDMs against spin-separated 3-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   5. Check accuracy of spin-separated LASSCF 3-RDMs by tracing them down to 2-RDMs
#   6. Check all of the above for different sub-spins of LASSCF fragments

import unittest

from pyscf import scf, gto, df, lib, fci
from pyscf import mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.addons import las2cas_civec
import numpy as np

def generate_geom (natom = 8, dist = 1.0):
    """Generate H chain of natoms each with dist distance linearly"""
    coords = []
    for i in range(natom):
        coords.append(f"H 0.0 0.0 {i*dist}")
    return "\n".join(coords)

def generate_ci_rdm123s (ci,nelec):
    nelec_sum = nelec[0][0]+nelec[0][1]
    norb = nelec_sum
    rdm1_fci, rdm2_fci, rdm3_fci = fci.rdm.make_dm123 ('FCI3pdm_kern_sf',ci,ci, norb, nelec_sum)
    rdm1_no, rdm2_no, rdm3_no = fci.rdm.reorder_dm123 (rdm1_fci, rdm2_fci, rdm3_fci) #normal ordered RDMs

    rdm1s, rdm2s, rdm3s = fci.direct_spin1.make_rdm123s(np.array(ci), norb, nelec[0])

    return rdm1_no, rdm2_no, rdm3_no, rdm1s, rdm2s, rdm3s

def mult_frags(nelesub, norbsub, charge=None, spin_sub=None, frag_atom_list=None, density_fit=False):
    """Used for checking systems with more than three LAS fragments to see if the 3-rdms are generated properly, here for H8 with 4 fragments."""
    xyz = generate_geom (natom = 8, dist = 1.0)
    mol = gto.M(atom=xyz, basis='sto-3g', charge=charge, verbose=0)
    mf = scf.RHF(mol).run()
    
    if spin_sub is None:
        spin_sub = (1,1,1,1)
    else:
        spin_sub = spin_sub
    
    if frag_atom_list is None: frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
    
    las = LASSCF (mf, nelesub, norbsub, spin_sub=spin_sub)
    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)

    rdm1_las = las.make_casdm1()
    rdm1s_las = las.make_casdm1s()
    rdm2_las = las.make_casdm2()
    rdm2s_las = las.make_casdm2s()
    rdm3_las = las.make_casdm3()
    rdm3s_las = las.make_casdm3s()

    ci, nelec = las2cas_civec(las)
    rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s = generate_ci_rdm123s(ci,nelec) # normal ordered RDMs

    return rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s

class KnownValues(unittest.TestCase):

    def trace_down_test(self, rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s):
        """ Test for tracing down spin-sep 3-RDMs down to 2-RDMs"""
        nelec = 8
        fac = 1. / (nelec//2-1)
        rdm2s_1a1 = np.einsum('ijkk->ij',rdm2s_las[0])*fac
        rdm2s_1a2 = np.einsum('kkij->ij',rdm2s_las[0])*fac
        rdm2s_1a3 = np.einsum('ijkk->ij',rdm2s_las[1])*(1.0/(nelec//2))
        rdm2s_1b1 = np.einsum('ijkk->ij',rdm2s_las[2])*fac
        rdm2s_1b2 = np.einsum('kkij->ij',rdm2s_las[2])*fac
        rdm2s_1b3 = np.einsum('ijkk->ij',rdm2s_las[1])*(1.0/(nelec//2))

        self.assertAlmostEqual (lib.fp (rdm2s_1a1), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1a2), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1a3), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b1), lib.fp (rdm1s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b2), lib.fp (rdm1s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b3), lib.fp (rdm1s_las[1]), 12)

        # Tracing down ss 3-RDM to 2-RDM
        nelec = 8

        rdm3s_2a1 = np.einsum('ijklmm->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a2 = np.einsum('ijmmkl->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a3 = np.einsum('mmijkl->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a4 = np.einsum('ijklmm->ijkl',rdm3s_las[1])*(1.0/(nelec//2))

        rdm3s_2b1 = np.einsum('ijklmm->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b2 = np.einsum('ijmmkl->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b3 = np.einsum('mmijkl->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b4 = np.einsum('mmijkl->ijkl',rdm3s_las[2])*(1.0/(nelec//2))

        rdm3s_2ab1 = np.einsum('ijmmkl->ijkl',rdm3s_las[1])*(1.0/(nelec//2-1))
        rdm3s_2ab2 = np.einsum('mmijkl->ijkl',rdm3s_las[1])*(1.0/(nelec//2-1))
        rdm3s_2ab3 = np.einsum('ijklmm->ijkl',rdm3s_las[2])*(1.0/(nelec//2-1))
        rdm3s_2ab4 = np.einsum('ijmmkl->ijkl',rdm3s_las[2])*(1.0/(nelec//2-1))

        self.assertAlmostEqual (lib.fp (rdm3s_2a1), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a2), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a3), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a4), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b1), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b2), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b3), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b4), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab1), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab2), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab3), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab4), lib.fp (rdm2s_las[1]), 12)

    def test_rdm3_h8_sto3g_spin1(self):
        """Spin cas 1"""
        spin_sub = (1,1,1,1)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)
        rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s = mult_frags(nelesub, norbsub, charge=None, spin_sub=spin_sub, density_fit=False)

        # Direct tests with ci generated RDMs
        self.assertAlmostEqual (lib.fp (rdm1_las), lib.fp (rdm1), 12)
        self.assertAlmostEqual (lib.fp (rdm1s_las), lib.fp (rdm1s), 12)
        self.assertAlmostEqual (lib.fp (rdm2_las), lib.fp (rdm2), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_las), lib.fp (rdm2s), 12)
        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_las), lib.fp (rdm3s), 12)

        # Tracing down tests
        nelec = 8
        fac = 1. / (nelec//2-1)
        rdm2s_1a1 = np.einsum('ijkk->ij',rdm2s_las[0])*fac
        rdm2s_1a2 = np.einsum('kkij->ij',rdm2s_las[0])*fac
        rdm2s_1a3 = np.einsum('ijkk->ij',rdm2s_las[1])*(1.0/(nelec//2))
        rdm2s_1b1 = np.einsum('ijkk->ij',rdm2s_las[2])*fac
        rdm2s_1b2 = np.einsum('kkij->ij',rdm2s_las[2])*fac
        rdm2s_1b3 = np.einsum('ijkk->ij',rdm2s_las[1])*(1.0/(nelec//2))

        self.assertAlmostEqual (lib.fp (rdm2s_1a1), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1a2), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1a3), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b1), lib.fp (rdm1s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b2), lib.fp (rdm1s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b3), lib.fp (rdm1s_las[1]), 12)

        # Tracing down ss 3-RDM to 2-RDM
        nelec = 8

        rdm3s_2a1 = np.einsum('ijklmm->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a2 = np.einsum('ijmmkl->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a3 = np.einsum('mmijkl->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a4 = np.einsum('ijklmm->ijkl',rdm3s_las[1])*(1.0/(nelec//2))

        rdm3s_2b1 = np.einsum('ijklmm->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b2 = np.einsum('ijmmkl->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b3 = np.einsum('mmijkl->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b4 = np.einsum('mmijkl->ijkl',rdm3s_las[2])*(1.0/(nelec//2))

        rdm3s_2ab1 = np.einsum('ijmmkl->ijkl',rdm3s_las[1])*(1.0/(nelec//2-1))
        rdm3s_2ab2 = np.einsum('mmijkl->ijkl',rdm3s_las[1])*(1.0/(nelec//2-1))
        rdm3s_2ab3 = np.einsum('ijklmm->ijkl',rdm3s_las[2])*(1.0/(nelec//2-1))
        rdm3s_2ab4 = np.einsum('ijmmkl->ijkl',rdm3s_las[2])*(1.0/(nelec//2-1))

        self.assertAlmostEqual (lib.fp (rdm3s_2a1), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a2), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a3), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a4), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b1), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b2), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b3), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b4), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab1), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab2), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab3), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab4), lib.fp (rdm2s_las[1]), 12)

    def test_rdm3_h8_sto3g_spin2(self):
        """Spin cas 2"""
        spin_sub = (1,3,1,3)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)

        rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s = mult_frags(nelesub, norbsub, charge=None, spin_sub=spin_sub, density_fit=False)

        # Direct tests with ci generated RDMs
        self.assertAlmostEqual (lib.fp (rdm1_las), lib.fp (rdm1), 12)
        self.assertAlmostEqual (lib.fp (rdm1s_las), lib.fp (rdm1s), 12)
        self.assertAlmostEqual (lib.fp (rdm2_las), lib.fp (rdm2), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_las), lib.fp (rdm2s), 12)
        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_las), lib.fp (rdm3s), 12)
        # Tracing down tests
        self.trace_down_test(rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s)


    def test_rdm3_h8_sto3g_spin3(self):
        """Spin cas 3"""
        spin_sub = (1,3,3,3)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)

        rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s = mult_frags(nelesub, norbsub, charge=None, spin_sub=spin_sub, density_fit=False)

        # Direct tests with ci generated RDMs
        self.assertAlmostEqual (lib.fp (rdm1_las), lib.fp (rdm1), 12)
        self.assertAlmostEqual (lib.fp (rdm1s_las), lib.fp (rdm1s), 12)
        self.assertAlmostEqual (lib.fp (rdm2_las), lib.fp (rdm2), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_las), lib.fp (rdm2s), 12)
        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_las), lib.fp (rdm3s), 12)
        # Tracing down tests
        self.trace_down_test(rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s)

    def test_rdm3_h8_sto3g_spin4(self):
        """Spin cas 4"""
        spin_sub = (3,3,3,3)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)

        rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s = mult_frags(nelesub, norbsub, charge=None, spin_sub=spin_sub, density_fit=False)

        # Direct tests with ci generated RDMs
        self.assertAlmostEqual (lib.fp (rdm1_las), lib.fp (rdm1), 12)
        self.assertAlmostEqual (lib.fp (rdm1s_las), lib.fp (rdm1s), 12)
        self.assertAlmostEqual (lib.fp (rdm2_las), lib.fp (rdm2), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_las), lib.fp (rdm2s), 12)
        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_las), lib.fp (rdm3s), 12)
        # Tracing down tests
        self.trace_down_test(rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s)

if __name__ == "__main__":
    print("Full Tests for LASSCF 3-RDMs")
    unittest.main()
