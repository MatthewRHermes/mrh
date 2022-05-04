import numpy as np
from pyscf import gto, scf, df, fci
from pyscf.fci.addons import fix_spin_
from mrh.my_pyscf import mcpdft
#from mrh.my_pyscf.fci import csf_solver
import unittest

def diatomic (atom1, atom2, r, fnal, basis, ncas, nelecas, nstates, charge=None, spin=None,
  symmetry=False, cas_irrep=None, density_fit=False):
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format (atom1, atom2, r)
    mol = gto.M (atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=0, output='/dev/null')
    mf = scf.RHF (mol)
    if density_fit: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mf.kernel ()
    mc = mcpdft.CASSCF (mf, fnal, ncas, nelecas, grids_level=9)
    #if spin is not None: smult = spin+1
    #else: smult = (mol.nelectron % 2) + 1
    #mc.fcisolver = csf_solver (mol, smult=smult)
    if spin is None: spin = mol.nelectron%2
    ss = spin*(spin+2)*0.25
    mc = mc.multi_state ([1.0/float(nstates),]*nstates, 'cms')
    mc.fix_spin_(ss=ss, shift=1)
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

    def test_h2_cms3ftlda22_sto3g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)
        e_ref = [-1.02544144, -0.44985771, -0.23390995]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (3):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_h2_cms2ftlda22_sto3g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        e_ref = [-1.11342858, -0.50064433]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_h2_cms3ftlda22_631g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 3)
        e_ref = [-1.08553117, -0.69136123, -0.49602992]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (3):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_h2_cms2ftlda22_631g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 2)
        e_ref = [-1.13120015, -0.71600911]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_lih_cms2ftlda44_sto3g (self):
        e = diatomic ('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})
        e_ref = [-7.86001566, -7.71804507]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_lih_cms2ftlda22_sto3g (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        e_ref = [-7.77572652, -7.68950326]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_lih_cms2ftlda22_sto3g_df (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2, density_fit=True)
        e_ref = [-7.776307, -7.689764]
        # Reference values from this program
        for i in range (2):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_lih_cms3ftlda22_sto3g (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)
        e_ref = [-7.79692534, -7.64435032, -7.35033371]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (3):
         with self.subTest (state=i):
            self.assertAlmostEqual (e[i], e_ref[i], 5)


if __name__ == "__main__":
    print("Full Tests for CMS-PDFT energies of diatomic molecules")
    unittest.main()






