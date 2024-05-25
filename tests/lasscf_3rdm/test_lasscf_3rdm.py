#!/usr/bin/env python
#
# Author: Shreya Verma <shreyav@uchicago.edu>

# The following tests are broken down into a couple of different categories.
#   1. Check accuracy of LASSCF analytical gradients with single fragment to CASSCF gradients for a diatomic molecule.
#   2. Check the implemmentation as scanner object.

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

def generate_ci_rdm123 (ci,nelec):
    nelec = nelec[0][0]+nelec[0][1]
    rdm1_fci, rdm2_fci, rdm3_fci = fci.rdm.make_dm123 ('FCI3pdm_kern_sf',ci,ci, nelec, nelec)
    rdm1_no, rdm2_no, rdm3_no = fci.rdm.reorder_dm123 (rdm1_fci, rdm2_fci, rdm3_fci) #normal ordered RDMs
    return rdm3_no

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
    
    rdm3_las = las.make_casdm3()

    ci, nelec = las2cas_civec(las)
    rdm3_no = generate_ci_rdm123(ci,nelec)

    return rdm3_las, rdm3_no

class KnownValues(unittest.TestCase):

    def test_rdm3_h8_sto3g_spin1(self):
        """Spin case 1"""
        spin_sub = (1,1,1,1)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)
        rdm3_las, rdm3_no = mult_frags(nelesub, norbsub, charge=None, spin_sub=None, density_fit=False)
        
        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3_las), 12)

    def test_rdm3_h8_sto3g_spin2(self):
        """Spin case 2"""
        spin_sub = (1,3,1,3)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)
        rdm3_las, rdm3_no = mult_frags(nelesub, norbsub, charge=None, spin_sub=spin_sub, density_fit=False)

        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3_las), 12)

    def test_rdm3_h8_sto3g_spin3(self):
        """Spin case 3"""
        spin_sub = (1,3,3,3)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)
        rdm3_las, rdm3_no = mult_frags(nelesub, norbsub, charge=None, spin_sub=spin_sub, density_fit=False)

        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3_las), 12)

    def test_rdm3_h8_sto3g_spin4(self):
        """Spin case 4"""
        spin_sub = (3,3,3,3)
        frag_atom_list = ((0, 1), (2, 3), (4, 5), (6,7))
        nelesub = (2,2,2,2)
        norbsub = (2,2,2,2)
        rdm3_las, rdm3_no = mult_frags(nelesub, norbsub, charge=None, spin_sub=None, density_fit=False)

        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3_las), 12)

if __name__ == "__main__":
    print("Full Tests for LASSCF 3-RDMs")
    unittest.main()
