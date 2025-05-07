import unittest
import numpy as np
from pyscf import gto, scf
from DMET.my_pyscf.dmet import runDMET

'''
***** RHF Embedding *****
1. Consider all the atoms in embedding space
2. Consider few atoms in embedding space
3. Consider few atoms in embedding space with density fitting
***** ROHF Embedding *****
1. Consider all the atoms in embedding space
2. Consider few atoms in embedding space
3. Consider few atoms in embedding space with density fitting
'''

def get_mole1():
    mol = gto.Mole(basis='6-31G', spin=0, charge=0, verbose=0)
    mol.atom = '''
    S  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()
    return mol

def get_mole2():
    mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=0)
    mol.atom = '''
    P  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()
    return mol

class KnownValues(unittest.TestCase):

    # RHF Embedding
    def test_vanilla_rhf(self):
        mol = get_mole1()
        mf = scf.RHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_energy, core_energy = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[:2] # Considering all the atoms in embedding space
        e_check = dmet_energy + core_energy
        del mol, mf, dmet_energy, core_energy
        self.assertAlmostEqual(e_ref, e_check, 6)
        
    def test_dmet_rhf(self):
        mol = get_mole1()
        mf = scf.RHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_energy, core_energy = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[:2] # Only few atoms in embedding space
        e_check = dmet_energy + core_energy
        del mol, mf, dmet_energy, core_energy
        self.assertAlmostEqual(e_ref, e_check, 6)

    def test_dmet_rhf_with_density_fitting(self):
        mol = get_mole1()
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        e_ref = mf.e_tot
        dmet_energy, core_energy = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[:2]
        e_check = dmet_energy + core_energy
        del mol, mf, dmet_energy, core_energy
        self.assertAlmostEqual(e_ref, e_check, 6)
    
    # ROHF Embedding
    def test_vanilla_rohf(self):
        mol = get_mole2()
        mf = scf.ROHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_energy, core_energy = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[:2]
        e_check = dmet_energy + core_energy
        del mol, mf, dmet_energy, core_energy
        self.assertAlmostEqual(e_ref, e_check, 6)

    def test_dmet_rohf(self):
        mol = get_mole2()
        mf = scf.ROHF(mol)
        mf.kernel()
        e_ref = mf.e_tot
        dmet_energy, core_energy = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[:2]
        e_check = dmet_energy + core_energy
        del mol, mf, dmet_energy, core_energy
        self.assertAlmostEqual(e_ref, e_check, 6)
        
    def test_dmet_rohf_with_density_fitting(self):
        mol = get_mole2()
        mf = scf.ROHF(mol).density_fit()
        mf.kernel()
        e_ref = mf.e_tot
        dmet_energy, core_energy = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[:2]
        e_check = dmet_energy + core_energy
        del mol, mf, dmet_energy, core_energy
        self.assertAlmostEqual(e_ref, e_check, 6)

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
