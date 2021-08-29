import numpy as np
from pyscf import gto, scf
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
import unittest

def diatomic (atom1, atom2, r, fnal, basis, ncas, nelecas, nstates, charge=None, spin=None, symmetry=False, cas_irrep=None):
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format (atom1, atom2, r)
    mol = gto.M (atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, fnal, ncas, nelecas, grids_level=9)
    if spin is not None: smult = spin+1
    else: smult = (mol.nelectron % 2) + 1
    mc.fcisolver = csf_solver (mol, smult=smult)
    mc = mc.state_interaction ([1.0/float(nstates),]*nstates, 'cms')
    mc.conv_tol = mc.conv_tol_sarot = 1e-12
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep (cas_irrep)
    mc.kernel (mo)
    return mc.e_states

def tearDownModule():
    global diatomic
    del diatomic

class KnownValues(unittest.TestCase):

    def test_lih_cms2ftlda44_sto3g (self):
        e = diatomic ('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})
        e_ref = [-7.86001566, -7.71804507]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_lih_cms2ftlda44_sto3g (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        e_ref = [-7.77572652, -7.68950326]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 6)


if __name__ == "__main__":
    print("Full Tests for CMS-PDFT energies of diatomic molecules")
    unittest.main()






